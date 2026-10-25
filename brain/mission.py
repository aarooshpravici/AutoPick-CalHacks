"""M1 mission and UI adapter, with gated measured navigation."""
from __future__ import annotations

import argparse
import copy
from datetime import date
import json
import math
from pathlib import Path
import queue
import threading
import time

from arm.pick import pick_to_hand
from arm.sim import SimArm, DEFAULT_POSES
from base.driver import BaseDriver, FakeSerial
from brain.detections import FakeDetections, ReplayDetections, UdpDetections, validate
from perception.publish import address

ROOT = Path(__file__).resolve().parents[1]


def grasp_settings(config):
    """Validate selection before any hardware opens; the passive cup has no take sensor."""
    handoff = config.get("handoff", {})
    grasp = handoff.get("grasp", "claw")
    if grasp not in ("claw", "suction"):
        raise ValueError("handoff.grasp must be claw or suction")
    options = {}
    if grasp == "suction":
        options["grasp"] = grasp
        for key, default, maximum in (("suction_dwell_s", 2, 10), ("suction_hold_s", 4, 60)):
            value = handoff.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"handoff.{key} must be positive and at most {maximum} seconds")
            options[key] = value
    return grasp, options


CAMERA_RESUME_GRACE_S = 8   # after the arm cameras close, before Search counts the camera as lost


def arm_policy_settings(config):
    """The ACT checkpoint run after the manual arm-camera switch (A14), or None.

    With a policy the mission parks, then waits in ArmCamera until the person presses
    "Switch to arm camera": the colour detector releases the RealSense and the policy
    opens it as the grey IR webcam its demos were recorded with."""
    path = config.get("handoff", {}).get("arm_policy")
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError("handoff.arm_policy must be a checkpoint folder path or null")
    if config["handoff"].get("grasp", "claw") != "claw":
        raise ValueError("handoff.arm_policy needs grasp: claw (suction takes no policy)")
    address(config["perception"].get("control_udp", ""))   # raises on a bad host:port
    return path


def handoff_timeout(config):
    """Validate the pick-thread timeout up front so a bad value fails before any motion."""
    value = config.get("handoff", {}).get("timeout_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("handoff.timeout_s must be a positive, finite number of seconds")
    return value


class SimNavigation:
    """A navigation stand-in, not measured parking or a real search controller."""
    def __init__(self):
        self.search_ticks = self.approach_ticks = 0

    def search(self):
        self.search_ticks += 1
        # 24 simulated 30-degree turn/pause steps = two full turns.
        return self.search_ticks >= 120, (-40, 40) if self.search_ticks % 5 < 3 else (0, 0)

    def approach(self, detection):
        self.approach_ticks += 1
        return self.approach_ticks >= 5, (40, 40)


class Mission:
    def __init__(self, config, *, base, arm, source, poses=DEFAULT_POSES,
                 navigator_factory=SimNavigation, pick=pick_to_hand, log_dir=None,
                 period=.1, simulated=True, arm_camera=None):
        self.grasp, self.pick_options = grasp_settings(config)
        self.arm_policy = arm_policy_settings(config)
        if self.arm_policy and arm_camera is None:
            from brain.arm_camera import ArmCamera, SimArmCamera
            arm_camera = SimArmCamera() if simulated else ArmCamera(config)
        self.arm_camera = arm_camera if self.arm_policy else None
        self._arm_go = False
        self.config, self.base, self.arm, self.source = config, base, arm, source
        self.poses, self.pick, self.navigator_factory = poses, pick, navigator_factory
        self.period = period
        self.simulated = simulated
        self._base_status = None
        self._limit_pending = None
        self._limit_ready = True
        self._boundary_started = None
        self._boundary_frame = None
        self._lock = threading.Lock()  # Never held during file, serial or arm I/O.
        self._closed = threading.Event()
        self._wake_io = threading.Event()
        self._token = threading.Event()
        self._generation = 0
        self._state, self._message, self._item = "Idle", "Ready (simulation)" if simulated else "Ready", None
        self._detection = self._last_pick = None
        self._seen = self._attempts = self._researches = 0
        self._ready_at = None  # Monotonic time of the last arm-ready reading.
        self._arm_epoch = 0  # Bumped when a pick starts, so an older reading cannot count as ready.
        self._last_frame = None
        self._last_camera = time.monotonic()
        self._last_seen = self._last_camera
        self._failed_at = 0
        self._navigator = navigator_factory()
        self._pick_thread = None
        self._result = None
        self._desired = None
        self._stop_needed = False
        self._stop_ticket = self._stop_completed = 0
        self._parking_stop_ticket = None
        self._io_started = time.monotonic()
        self._io_error = None
        self._log_error = None
        self._events = queue.SimpleQueue()
        directory = Path(log_dir) if log_dir else ROOT / "data/logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / f"{date.today().isoformat()}.jsonl"
        # Open before starting workers; logging failure cannot leave a mission running.
        self._log_file = self.log_path.open("a")
        self._logger = threading.Thread(target=self._write_log, daemon=True, name="mission-log")
        self._io = threading.Thread(target=self._base_io, daemon=True, name="mission-base")
        self._loop = threading.Thread(target=self._run, daemon=True, name="mission-loop")
        self._logger.start(); self._io.start(); self._loop.start()
        self._event("started", simulated=simulated)

    def _event(self, kind, **fields):
        self._events.put(dict(t=time.time(), event=kind, **fields))

    def _write_log(self):
        try:
            while True:
                record = self._events.get()
                if record is None:
                    break
                self._log_file.write(json.dumps(record, allow_nan=False) + "\n")
                self._log_file.flush()
        except Exception as error:
            self._log_error = str(error)
        finally:
            self._log_file.close()

    def _state_to(self, state, message):
        if state != self._state or message != self._message:
            self._state, self._message = state, message
            self._event("state", state=state, message=message, item=self._item)
        if state == "Failed":
            self._failed_at = time.monotonic()

    def _halt(self):
        self._desired = None
        self._stop_needed = True
        self._stop_ticket += 1
        self._wake_io.set()

    def request(self, item, hand, *, max_distance_ft=None):
        if max_distance_ft is not None and (isinstance(max_distance_ft, bool) or
                not isinstance(max_distance_ft, (int, float)) or
                not math.isfinite(max_distance_ft) or not 0 < max_distance_ft <= 20):
            raise ValueError("distance must be greater than 0 and at most 20 feet")
        if item not in self.config["items"] or hand not in ("seated", "standing"):
            raise ValueError("unsupported item or hand preset")
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("brain closed")
            if self._io_error or self._log_error:
                raise RuntimeError("brain fault; reconnect before requesting")
            self._token.set()
            self._token = threading.Event()
            self._generation += 1
            self._item = item
            # A fresh budget is installed only after acknowledged STOP. Legacy calls
            # also get a bounded default, so an old exhausted budget cannot leak.
            self._limit_pending = (self._generation, (20 if max_distance_ft is None else max_distance_ft) * 304.8)
            self._limit_ready = False
            self._boundary_started = None
            self._seen = self._attempts = self._researches = 0
            self._ready_at = None
            self._last_frame = None
            self._last_camera = self._last_seen = time.monotonic()
            self._navigator = self.navigator_factory()
            self._parking_stop_ticket = None
            self._arm_go = False
            self._halt()
            self._state_to("Search", f"Looking for: {item}")
            self._event("request", item=item, hand=hand)

    def stop(self):
        with self._lock:
            self._token.set()
            self._generation += 1
            self._arm_go = False
            self._halt()
            self._state_to("Stopped", "Stopped; arm holding")
            self._event("stop")
            error = self._io_error
        if error:
            raise RuntimeError(f"STOP FAILED: {error}")

    def arm_camera_go(self):
        """The page's "Switch to arm camera" button: valid only while parked in ArmCamera."""
        with self._lock:
            if self._state != "ArmCamera":
                raise RuntimeError("not parked and waiting for the arm camera")
            self._arm_go = True
            self._event("arm_camera_go")

    def status(self):
        with self._lock:
            return dict(state=self._state, item=self._item,
                        message=f"STOP FAILED: {self._io_error}" if self._io_error else self._message,
                        detection=copy.deepcopy(self._detection), lift_mm=None, tray_g=None,
                        last_pick=copy.deepcopy(self._last_pick), cameras=[], t=time.time())

    def thumbnail(self, name):
        return None

    def _arm_ready(self):
        position = self.arm.read()
        frozen = self.config['arm'].get('frozen_joints', [])
        joints = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
        return all(joint in position and math.isfinite(position[joint]) and
                   abs(position[joint] - self.poses['ready'][joint]) <= 5
                   for joint in joints if joint not in frozen)

    def _arm_refuse(self, generation, message):
        with self._lock:
            if generation == self._generation:
                self._token.set()
                self._limit_pending = None
                self._halt()
                self._state_to("Stopped", message)

    def _arm_gate(self, generation, epoch, max_age):
        """True only with the arm at taught ready, read at most max_age s ago; else stop the request."""
        now = time.monotonic()
        ready_at = self._ready_at
        if max_age and ready_at is not None and now - ready_at <= max_age:
            return True
        try:
            ready = self._arm_ready()
        except Exception as error:  # An arm fault is not a base/control-connection failure.
            self._arm_refuse(generation, f"Arm unreadable; base motion refused ({str(error)[:80]})")
            return False
        with self._lock:
            self._ready_at = now if ready and epoch == self._arm_epoch else None
        if not ready:
            self._arm_refuse(generation, "Arm is not at taught ready pose; return it safely before driving")
        return ready

    def _base_io(self):
        while not self._closed.is_set():
            self._wake_io.wait(.05)
            self._wake_io.clear()
            with self._lock:
                halt = self._stop_needed
                stop_ticket = self._stop_ticket
                self._stop_needed = False
                desired = self._desired
                self._desired = None
                generation = self._generation
                cancelled = self._token.is_set()
                pending_limit = self._limit_pending
                arm_busy = self._pick_thread is not None and self._pick_thread.is_alive()
                epoch = self._arm_epoch
                if arm_busy:
                    self._ready_at = None
            try:
                if halt:
                    self.base.stop_and_wait()
                    self._event("serial", command="STOP")
                    with self._lock:
                        self._stop_completed = stop_ticket
                elif pending_limit and pending_limit[0] == generation and not cancelled and not arm_busy:
                    if self._arm_gate(generation, epoch, 0):  # Read only, after STOP, before any base motion.
                        self.base.set_travel_limit(pending_limit[1])
                        with self._lock:
                            if generation == self._generation and not self._token.is_set():
                                self._limit_pending = None
                                self._limit_ready = True
                        self._event("serial", command="LIMIT", mm=pending_limit[1])
                elif desired and desired[0] == generation and not cancelled and not arm_busy:
                    if not any(desired[1]) or self._arm_gate(generation, epoch, .2):
                        self.base.drive(*desired[1])
                        self._event("serial", command="DRIVE", speeds=desired[1])
                status = self.base.status(timeout=0)
                with self._lock:
                    self._base_status = status
                if time.monotonic() - status.received_at > .3 or status.estop:
                    raise RuntimeError("stale base status or E-stop")
            except Exception as error:
                if isinstance(error, TimeoutError) and time.monotonic() - self._io_started < .3:
                    continue
                with self._lock:
                    self._io_error = str(error)
                    self._token.set()
                    self._desired = None
                    self._state_to("Failed", f"Control connection failed: {error}; use E-stop")
                try:
                    self.base.stop()
                except Exception:
                    pass
                return

    def _drive(self, speeds):
        limit = self.config["base"]["max_speed_mm_s"]
        scale = max(1, max(abs(value) for value in speeds) / limit)
        self._desired = (self._generation, tuple(value / scale for value in speeds))
        self._wake_io.set()

    def _start_pick(self, generation, token):
        def on_state(state):
            with self._lock:
                if generation == self._generation and not token.is_set():
                    self._state_to({"pick": "Pick", "present": "Present", "wait_for_take": "WaitForTake"}[state],
                                   {"pick": "Picking up", "present": "Presenting", "wait_for_take": "Please take the item"}[state])
        options = dict(self.pick_options)
        camera_notes = []
        try:
            if self.arm_camera is not None:
                policy = self.arm_camera.open(log=lambda message: self._event("arm", message=message))
                if policy is not None:
                    options["policy"] = policy
            result = self.pick(self.arm, self.poses,
                               speed_deg_s=(self.config["arm"]["move_speed_deg_s"] if self.simulated else
                                            min(25, self.config["arm"]["move_speed_deg_s"])),
                               handoff_timeout_s=self.config["handoff"]["timeout_s"],
                               handoff_grip=self.config["handoff"].get("grip", "squeeze"),
                               stop_event=token, on_state=on_state,
                               log=lambda message: self._event("arm", message=message), **options)
            if self.grasp == "suction" and result.get("result") == "taken":
                result = dict(result, result="unverified", take_verified=False)
                result["detail"] = "Timed suction sequence finished; seal and take were not sensed. " + result.get("detail", "")
        except Exception as error:
            result = dict(result="stuck", detail=str(error), grasp="policy" if self.arm_camera else "scripted",
                          attempts=0, seconds=0)
        finally:
            if self.arm_camera is not None:
                try:
                    camera_notes = self.arm_camera.close()
                except Exception as error:  # the result must still be reported
                    camera_notes = [f"arm camera close failed: {error}"]
        if camera_notes:
            result = dict(result, camera_notes=camera_notes)
        with self._lock:
            self._result = generation, result
        self._event("pick_result", generation=generation, result=result)

    def _tick(self):
        message = self.source.poll()
        if message is not None:
            validate(message)
            self._event("detection", message=message)
        start_pick = None
        now = time.monotonic()
        with self._lock:
            if self._log_error:
                self._token.set(); self._halt()
                self._state_to("Failed", "Logging failed")
                return
            if message is not None and -0.5 <= time.time() - message["t"] <= 1:
                key = (message["frame"], message["t"])
                if key != self._last_frame:
                    self._last_frame = key
                    self._last_camera = now
                    self._detection = max((item for item in message["items"]
                                           if item["label"] == self._item and item["conf"] >= .6),
                                          key=lambda item: item["conf"], default=None)
                    self._seen = self._seen + 1 if self._detection else 0
                    if self._detection:
                        self._last_seen = now
            if self._result is not None:
                generation, result = self._result
                self._result = None
                if generation == self._generation and not self._token.is_set():
                    self._last_pick = result
                    if result["result"] == "taken":
                        self._state_to("Idle", "Item taken")
                    elif result["result"] == "unverified":
                        self._halt()
                        self._state_to("Idle", "Suction sequence finished; visually confirm pickup and take")
                    elif result["result"] == "dropped" and self._attempts < 3:
                        self._seen = 0
                        self._navigator = self.navigator_factory()
                        self._parking_stop_ticket = None
                        if self.arm_camera is not None:   # the colour detector needs a few seconds to reopen
                            self._last_camera = time.monotonic() + CAMERA_RESUME_GRACE_S
                        self._state_to("Search", "Dropped; searching again")
                    elif result["result"] == "stopped":
                        self._token.set(); self._halt()
                        self._state_to("Stopped", "Arm stopped and holding")
                    else:
                        self._halt()
                        self._state_to("Failed", f"Pick failed: {result['result']}")
            if hasattr(self._navigator, "observe"):
                age = float("inf") if self._last_frame is None else time.time() - self._last_frame[1]
                self._navigator.observe(self._last_frame, age, self._base_status)
            if self._state == "BoundaryCheck":
                if not self._boundary_stopped and self._stop_completed >= self._stop_ticket:
                    self._boundary_stopped = True
                    self._boundary_started = time.time()
                    self._boundary_frame = self._last_frame
                    return
                # Demand a frame captured after stopping, not a cached detection.
                fresh = (self._boundary_stopped and self._last_frame is not None
                         and self._last_frame != self._boundary_frame
                         and self._last_frame[1] >= self._boundary_started
                         and 0 <= time.time() - self._last_frame[1] <= .3)
                if fresh or time.time() - self._boundary_started > 2:
                    result = ("Item seen, but travel limit reached; reposition before another request"
                              if fresh and self._detection else
                              "Item not found at travel limit; reposition before another request" if fresh else
                              "Travel limit reached; no fresh camera frame")
                    self._token.set()
                    self._state_to("Stopped", result)
                return
            if self._state in ("Search", "Approach"):
                if not self._limit_ready:
                    return
                if self.base.limit_reached:
                    self._halt()
                    self._boundary_started = time.time()
                    self._boundary_frame = self._last_frame
                    self._boundary_stopped = False
                    self._state_to("BoundaryCheck", "Travel limit reached; stopped and checking camera")
                    return
                if self._base_status is None:
                    self._halt()
                    return
                if self._pick_thread is not None and self._pick_thread.is_alive():
                    self._halt()
                    return  # A replaced mission must finish holding the arm first.
                if now - self._last_camera > 1:
                    self._halt()
                    self._state_to("Failed", "Camera lost")
                elif self._state == "Search":
                    if self._seen >= 5:
                        self._halt()
                        self._state_to("Approach", "Approaching (simulated parking)" if self.simulated else "Approaching measured pickup spot")
                    else:
                        exhausted, speeds = self._navigator.search()
                        if exhausted:
                            self._halt(); self._state_to("Failed", "Two simulated turns; item not found" if self.simulated else "Search exhausted; item not found")
                        else:
                            self._drive(speeds)
                elif now - self._last_seen > 1:
                    self._halt(); self._seen = 0
                    if self._researches >= 3:  # Rebuilt navigators reset their own budgets; bound it here.
                        self._state_to("Failed", "Item repeatedly lost; stopping")
                        return
                    self._researches += 1
                    self._navigator = self.navigator_factory()
                    self._parking_stop_ticket = None
                    self._state_to("Search", "Item lost; searching again")
                elif self._detection is None:
                    self._parking_stop_ticket = None
                    self._halt()
                else:
                    parked, speeds = self._navigator.approach(self._detection)
                    if parked:
                        if self._parking_stop_ticket is None:
                            self._halt()
                            self._parking_stop_ticket = self._stop_ticket
                        elif self._stop_completed >= self._parking_stop_ticket:
                            if self.arm_camera is not None:
                                self._arm_go = False
                                self._state_to("ArmCamera", "Parked. Press Switch to arm camera to start the arm")
                            else:
                                start_pick = self._begin_pick()
                    else:
                        self._parking_stop_ticket = None  # Any renewed approach needs a new stopped acknowledgment.
                        self._drive(speeds)
            elif self._state == "ArmCamera" and self._arm_go:
                # Base stays stopped: ArmCamera never drives, and STOP was acknowledged on parking.
                self._arm_go = False
                start_pick = self._begin_pick()
            elif self._state == "Failed" and not self._io_error and now - self._failed_at > 1:
                self._state_to("Idle", self._message)
        if start_pick:
            # The firmware STOP acknowledgment must precede arm work.
            self._pick_thread = threading.Thread(target=self._start_pick, args=start_pick,
                                                 daemon=True, name="mission-pick")
            self._pick_thread.start()

    def _begin_pick(self):
        """Caller holds the lock, after the parking STOP acknowledgment."""
        self._attempts += 1
        self._state_to("Pick", "Switching to the arm camera" if self.arm_camera else "Picking up")
        self._arm_epoch += 1
        self._ready_at = None
        return self._generation, self._token

    def _run(self):
        while not self._closed.is_set():
            start = time.monotonic()
            try:
                self._tick()
            except Exception as error:
                with self._lock:
                    self._token.set(); self._halt()
                    self._state_to("Failed", f"Mission error: {error}")
            self._closed.wait(max(0, self.period - (time.monotonic() - start)))

    def close(self):
        if self._closed.is_set():
            return
        try:
            self.stop()
        finally:
            self._closed.set(); self._wake_io.set()
            self._loop.join(1); self._io.join(1)
            if self._pick_thread:
                self._pick_thread.join(2)
            self.base.close()
            self.source.close()
            self.arm.close()
            self._events.put(None)
            self._logger.join(1)


def build(config, sim):
    grasp_settings(config)
    handoff_timeout(config)
    if not sim:
        return build_real(config)
    cfg = config["base"]
    base = BaseDriver(FakeSerial(max_speed_mm_s=cfg["max_speed_mm_s"]),
                      max_speed_mm_s=cfg["max_speed_mm_s"], max_lift_mm=1000)
    return Mission(config, base=base, arm=SimArm(), source=FakeDetections(labels=config["items"]))


def build_real(config):
    # Validate every prerequisite before opening any serial device.
    from arm.poses import load
    from perception.floor import FloorMap
    from brain.navigation import RealNavigation
    grasp, _ = grasp_settings(config)
    policy = arm_policy_settings(config)
    if policy is not None:
        from arm.pick import load_camera_indices
        from arm.policy import checkpoint_image_sizes
        folder = Path(policy) if Path(policy).is_absolute() else ROOT / policy
        if not (folder / "config.json").is_file():
            raise RuntimeError(f"handoff.arm_policy has no checkpoint config.json: {folder}")
        load_camera_indices(ROOT / "config/robot.yaml", list(checkpoint_image_sizes(folder)))
    handoff_timeout(config)
    cfg = config["base"]
    if cfg.get("wiring_confirmed") is not True or cfg.get("drive_calibrated") is not True:
        raise RuntimeError("Real M1 requires P4 and completed wiring/loaded-drive checks")
    arm_cfg = config["arm"]
    frozen = arm_cfg.get("frozen_joints")
    if not isinstance(frozen, list) or "wrist_roll" not in frozen:
        raise RuntimeError("faulty wrist_roll must remain frozen before real startup")
    for name, maximum in (("max_relative_target_deg", 10), ("move_speed_deg_s", 45)):
        value = arm_cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
            raise RuntimeError(f"invalid arm safety limit: {name}")
    port_name = cfg.get("serial_port")
    if not isinstance(port_name, str) or not port_name.startswith("/dev/") or "XXXX" in port_name:
        raise RuntimeError("configure the measured Uno R4 serial port")
    normalize = lambda p: p.replace("/dev/tty.", "/dev/cu.")
    if normalize(port_name) in [normalize(arm_cfg[key]) for key in ("follower_port", "leader_port")]:
        raise RuntimeError("R4 port must not be an arm port")
    floor = FloorMap.load(ROOT / "config/floor_homography.json",
                          colour_size=config["realsense"]["colour"], flip=config["realsense"]["flip"])
    if math.dist(floor.parking(), config["perception"]["parking_spot_floor_m"]) > 1e-6:
        raise RuntimeError("P4 parking target differs from robot.yaml")
    poses = load(ROOT / "config/poses.json", mode="suction" if grasp == "suction" else "hand")
    factory = lambda: RealNavigation(config, floor)
    factory()  # Validate navigation before ports are opened.
    base = arm = source = None
    try:
        import serial
        from arm.real import RealArm
        source = UdpDetections(config["perception"]["detections_udp"])
        port = serial.Serial(cfg["serial_port"], baudrate=cfg["baud"], timeout=.05, write_timeout=.1, exclusive=True)
        base = BaseDriver(port, max_speed_mm_s=cfg["max_speed_mm_s"], max_lift_mm=1000)
        base.status(timeout=3)
        base.synchronize()
        base.verify_firmware(cfg)
        arm_cfg = config["arm"]
        arm = RealArm(arm_cfg["follower_port"], arm_cfg["follower_id"],
                      float(arm_cfg["max_relative_target_deg"]), frozen=arm_cfg.get("frozen_joints", ()))
        return Mission(config, base=base, arm=arm, source=source, poses=poses,
                       navigator_factory=factory, simulated=False)
    except BaseException:
        for resource in (base, source, arm):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        raise


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.yaml")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--udp", action="store_true", help="consume P3 --fake messages on detections_udp")
    sources.add_argument("--replay", type=Path, help="replay saved 9.2 JSONL")
    parser.add_argument("--item", default="phone")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    source = (UdpDetections(config["perception"]["detections_udp"]) if args.udp else
              ReplayDetections(args.replay) if args.replay else FakeDetections())
    base = BaseDriver.from_config(args.config, fake=True)
    brain = Mission(config, base=base, arm=SimArm(), source=source)
    try:
        brain.request(args.item, "seated")
        end = time.monotonic() + args.timeout
        while time.monotonic() < end:
            state = brain.status()
            if state["state"] in ("Idle", "Failed", "Stopped"):
                print(json.dumps(state))
                return 0 if state["last_pick"] and state["last_pick"]["result"] == "taken" else 1
            time.sleep(.1)
        print(json.dumps(brain.status()))
        return 1
    finally:
        brain.close()


if __name__ == "__main__":
    raise SystemExit(main())
