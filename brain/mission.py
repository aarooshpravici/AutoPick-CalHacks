"""M1 simulated mission and UI Brain adapter. Real navigation awaits P4/B5/B6."""
from __future__ import annotations

import argparse
import copy
from datetime import date
import json
from pathlib import Path
import queue
import threading
import time

from arm.pick import pick_to_hand
from arm.sim import SimArm, DEFAULT_POSES
from base.driver import BaseDriver, FakeSerial
from brain.detections import FakeDetections, ReplayDetections, UdpDetections, validate

ROOT = Path(__file__).resolve().parents[1]


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
                 period=.1):
        self.config, self.base, self.arm, self.source = config, base, arm, source
        self.poses, self.pick, self.navigator_factory = poses, pick, navigator_factory
        self.period = period
        self._lock = threading.Lock()  # Never held during file, serial or arm I/O.
        self._closed = threading.Event()
        self._wake_io = threading.Event()
        self._token = threading.Event()
        self._generation = 0
        self._state, self._message, self._item = "Idle", "Ready (simulation)", None
        self._detection = self._last_pick = None
        self._seen = self._attempts = 0
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
        self._event("started", simulated=True)

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

    def request(self, item, hand):
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
            self._seen = self._attempts = 0
            self._last_frame = None
            self._last_camera = self._last_seen = time.monotonic()
            self._navigator = self.navigator_factory()
            self._parking_stop_ticket = None
            self._halt()
            self._state_to("Search", f"Looking for: {item}")
            self._event("request", item=item, hand=hand)

    def stop(self):
        with self._lock:
            self._token.set()
            self._generation += 1
            self._halt()
            self._state_to("Stopped", "Stopped; arm holding")
            self._event("stop")
            error = self._io_error
        if error:
            raise RuntimeError(f"STOP FAILED: {error}")

    def status(self):
        with self._lock:
            return dict(state=self._state, item=self._item,
                        message=f"STOP FAILED: {self._io_error}" if self._io_error else self._message,
                        detection=copy.deepcopy(self._detection), lift_mm=None, tray_g=None,
                        last_pick=copy.deepcopy(self._last_pick), cameras=[], t=time.time())

    def thumbnail(self, name):
        return None

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
            try:
                if halt:
                    self.base.stop()
                    self._event("serial", command="STOP")
                    with self._lock:
                        self._stop_completed = stop_ticket
                elif desired and desired[0] == generation and not cancelled:
                    self.base.drive(*desired[1])
                    self._event("serial", command="DRIVE", speeds=desired[1])
                status = self.base.status(timeout=0)
                if time.monotonic() - status.received_at > .3 or status.estop:
                    raise RuntimeError("stale base status or E-stop")
            except Exception as error:
                if isinstance(error, TimeoutError) and time.monotonic() - self._io_started < .3:
                    continue
                with self._lock:
                    self._io_error = str(error)
                    self._token.set()
                    self._desired = None
                    self._state_to("Failed", "Base connection failed; use E-stop")
                try:
                    self.base.stop()
                except Exception:
                    pass
                return

    def _drive(self, speeds):
        self._desired = (self._generation, speeds)
        self._wake_io.set()

    def _start_pick(self, generation, token):
        def on_state(state):
            with self._lock:
                if generation == self._generation and not token.is_set():
                    self._state_to({"pick": "Pick", "present": "Present", "wait_for_take": "WaitForTake"}[state],
                                   {"pick": "Picking up", "present": "Presenting", "wait_for_take": "Please take the item"}[state])
        try:
            result = self.pick(self.arm, self.poses,
                               speed_deg_s=self.config["arm"]["move_speed_deg_s"],
                               handoff_timeout_s=self.config["handoff"]["timeout_s"],
                               handoff_grip=self.config["handoff"].get("grip", "squeeze"),
                               stop_event=token, on_state=on_state,
                               log=lambda message: self._event("arm", message=message))
        except Exception as error:
            result = dict(result="stuck", detail=str(error), grasp="scripted", attempts=0, seconds=0)
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
                    elif result["result"] == "dropped" and self._attempts < 3:
                        self._seen = 0
                        self._navigator = self.navigator_factory()
                        self._parking_stop_ticket = None
                        self._state_to("Search", "Dropped; searching again")
                    elif result["result"] == "stopped":
                        self._token.set(); self._halt()
                        self._state_to("Stopped", "Arm stopped and holding")
                    else:
                        self._halt()
                        self._state_to("Failed", f"Pick failed: {result['result']}")
            if self._state in ("Search", "Approach"):
                if self._pick_thread is not None and self._pick_thread.is_alive():
                    self._halt()
                    return  # A replaced mission must finish holding the arm first.
                if now - self._last_camera > 1:
                    self._halt()
                    self._state_to("Failed", "Camera lost")
                elif self._state == "Search":
                    if self._seen >= 5:
                        self._state_to("Approach", "Approaching (simulated parking)")
                    else:
                        exhausted, speeds = self._navigator.search()
                        if exhausted:
                            self._halt(); self._state_to("Failed", "Two simulated turns; item not found")
                        else:
                            self._drive(speeds)
                elif now - self._last_seen > 1:
                    self._halt(); self._seen = 0
                    self._navigator = self.navigator_factory()
                    self._parking_stop_ticket = None
                    self._state_to("Search", "Item lost; searching again")
                elif self._detection is None:
                    self._halt()
                else:
                    parked, speeds = self._navigator.approach(self._detection)
                    if parked:
                        if self._parking_stop_ticket is None:
                            self._halt()
                            self._parking_stop_ticket = self._stop_ticket
                        elif self._stop_completed >= self._parking_stop_ticket:
                            self._attempts += 1
                            self._state_to("Pick", "Picking up")
                            start_pick = self._generation, self._token
                    else:
                        self._drive(speeds)
            elif self._state == "Failed" and not self._io_error and now - self._failed_at > 1:
                self._state_to("Idle", self._message)
        if start_pick:
            # Stop the simulated base before arm work starts. No real-mode builder
            # is exposed until measured navigation and physical interlocks exist.
            self._pick_thread = threading.Thread(target=self._start_pick, args=start_pick,
                                                 daemon=True, name="mission-pick")
            self._pick_thread.start()

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
    if not sim:
        raise RuntimeError("Real M1 is gated on P4 calibration and B5/B6 navigation; use --sim")
    cfg = config["base"]
    base = BaseDriver(FakeSerial(max_speed_mm_s=cfg["max_speed_mm_s"]),
                      max_speed_mm_s=cfg["max_speed_mm_s"], max_lift_mm=1000)
    return Mission(config, base=base, arm=SimArm(), source=FakeDetections())


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
