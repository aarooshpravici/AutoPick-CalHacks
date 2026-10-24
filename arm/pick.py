"""Pick sequence: floor -> the person's hand, or (stretch) the tray (plan 9.4, 10.1, 10.6).

    result = pick_to_hand(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45,
                          handoff_timeout_s=60, stop_event=None, on_state=None, handoff_grip="squeeze")
    result = pick_to_tray(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45)

    python -m arm.pick run [--to hand|tray] [--policy DIR] [--pan DEG] [--roll DEG]
                           [--speed DEG_S] [--handoff-timeout S] [--handoff-grip squeeze|light]
                           [--dry-run]

The last printed line is `RESULT {json}`. Exit codes: 0 taken or in_tray, 1 error or
stopped, 2 missed, 3 dropped, 4 stuck, 5 timeout. The default is --to hand.

Hand-off (pick_to_hand): after the grasp the arm carries the object up to `lift`, then
to `present` at no more than PRESENT_SPEED_DEG_S (a person is near, plan 17), and holds
it there. The claw's reading on arriving there, holding, is `hold_pos`. handoff_grip
(robot.yaml handoff.grip, decision D9) sets the grip while it waits: "squeeze" keeps
the carry squeeze; "light" commands the claw only LIGHT_SQUEEZE past hold_pos (never
past the carry squeeze), so it presses less hard and is easier to pull from. Every
carry move squeezes in both modes; only the wait at `present` differs. When the person
pulls the object out, the claw closes onto its target. A read on the closed side, or at
least TAKE_CLOSE_UNITS closed from hold_pos (a light hold stops short of the closed
side), counts as closed; TAKE_READS closed reads in a row (at TAKE_POLL_HZ) are a take,
and the arm opens the claw and goes back to `ready` ("taken"). The person reaches only once
the arm has stopped (plan 17), so a claw that starts closing within TAKE_SETTLE_S of
arriving lost the object on its own ("dropped"), as does a claw that reads open for
TAKE_READS reads in a row. A slip later than that looks exactly like a take (known
limit, plan 10.7). If nobody takes it within handoff_timeout_s (robot.yaml
handoff.timeout_s), the arm goes back to `lift`, squeezing again ("timeout"; "dropped"
if the claw no longer holds it there). The moves away from `present` run at the same
lower speed.
on_state, if given, is called with "pick" at the start, "present" as the move to
`present` starts and "wait_for_take" once the arm is there and still holding.

`arm` is anything with read() -> {joint: value}, command({joint: value}) and close():
arm.sim.SimArm or arm.real.RealArm. If it has a `clock` (the sim does), timing runs
on it; otherwise on the wall clock.

`policy` (task T3) is an object with step(state) -> {joint: target} for all six
joints, and optionally reset(); or an ACT checkpoint folder, which load_policy()
opens with the robot.yaml webcams (arm/policy.py). It drives from `ready`, one step
per frame at RATE_HZ (the rate the demos were recorded at), until it asks for a
closed claw and the claw has stood still short of closed for 8 frames; then the
claw is squeezed shut, checked like a scripted grasp, and the scripted carry takes
over. A claw standing still fully closed for 8 frames closed on nothing (a missed
attempt). The 20 s time limit applies to each attempt; when it runs out the claw
is closed and checked the same way.

Speed: config/robot.yaml `arm.move_speed_deg_s` is the ceiling (plan 9.1, 17). The
command line runs at that speed by default and refuses a larger --speed; first real
runs use --speed 25. pick_to_hand() and pick_to_tray() themselves also refuse anything
above the plan's 45 deg/s arm limit (MAX_SPEED_DEG_S), for callers that skip
load_arm_config().
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import numbers
import signal
import sys
import threading
import time
from pathlib import Path

from . import poses as poses_mod
from .poses import ARM_JOINTS, REPO_ROOT

RATE_HZ = 30
GRIP_MARGIN = 4.0               # claw holds something: at least this far above taught closed
SQUEEZE = 5.0                   # while carrying, command the claw this far past closed
TOLERANCE_DEG = {"shoulder_pan": 4.0, "shoulder_lift": 8.0, "elbow_flex": 8.0,
                 "wrist_flex": 10.0, "wrist_roll": 6.0}
STUCK_AFTER_S = 2.0             # a joint still not there this long after the move ends
GRIP_SETTLE_S = 0.25            # claw counts as stopped after this long without moving
GRIP_TIMEOUT_S = 1.5
ATTEMPTS = 2
POLICY_HOLD_FRAMES = 8
POLICY_TIME_LIMIT_S = 20.0      # per attempt
GRIP_STILL_DEG = 0.3            # policy mode: claw moved less than this since the last frame = stopped
MAX_SPEED_DEG_S = 45.0          # plan section 17 arm limit; robot.yaml move_speed_deg_s is the working ceiling
PRESENT_SPEED_DEG_S = 25.0      # plan section 17: moves to and from `present` are near a person
TAKE_POLL_HZ = 20               # claw reads while waiting for the take
TAKE_READS = 6                  # reads in a row (about 0.3 s) that decide a take or a drop
TAKE_SETTLE_S = 1.0             # nobody reaches this soon after the arm stops: closing then is a slip
HANDOFF_GRIPS = ("squeeze", "light")    # grip while waiting at `present` (robot.yaml handoff.grip, D9)
LIGHT_SQUEEZE = 6.0             # light grip: command the claw this far past where it holds the object
TAKE_CLOSE_UNITS = 4.0          # closed this far from where it held the object = closed (below LIGHT_SQUEEZE)
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"

EXIT_CODES = {"taken": 0, "in_tray": 0, "stopped": 1, "error": 1, "missed": 2, "dropped": 3, "stuck": 4,
              "timeout": 5}


class RealClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


class _Abort(Exception):
    def __init__(self, result: str, detail: str):
        super().__init__(detail)
        self.result = result
        self.detail = detail


def _is_number(value) -> bool:
    """A finite real number (numpy scalars included), not a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def claw_holding(gripper: float, g_open: float, g_closed: float, margin: float = GRIP_MARGIN) -> bool:
    """True when the claw stopped short of closed (on an object) and isn't open."""
    d = 1.0 if g_open > g_closed else -1.0
    return (gripper - g_closed) * d >= margin and (g_open - gripper) * d >= 1.0


class _Picker:
    def __init__(self, arm, poses, pan_deg, roll_deg, speed_deg_s, stop_event, log):
        self.arm = arm
        self.poses = poses
        self.pan = pan_deg
        self.roll = roll_deg
        self.speed = speed_deg_s
        self.stop_event = stop_event
        self.log = log
        self.clock = getattr(arm, "clock", None) or RealClock()
        self.dt = 1.0 / RATE_HZ
        self.g_open = poses["gripper_open"]
        self.g_closed = poses["gripper_closed"]
        self.dir = 1.0 if self.g_open > self.g_closed else -1.0
        self.g_squeeze = self.g_closed - self.dir * SQUEEZE
        self.joint_cmd = None
        self.grip_cmd = None

    # --- low level -----------------------------------------------------------

    def _send(self, joints: dict, grip: float) -> None:
        self.joint_cmd = dict(joints)
        self.grip_cmd = grip
        self.arm.command({**joints, "gripper": grip})

    def _hold(self) -> None:
        """Hold the joints where they are; keep the claw's current target (keeps squeezing)."""
        pos = self.arm.read()
        grip = self.grip_cmd if self.grip_cmd is not None else pos["gripper"]
        self._send({j: pos[j] for j in ARM_JOINTS}, grip)

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            self._hold()
            raise _Abort("stopped", "STOP received; arm holding")

    def holding(self) -> bool:
        return claw_holding(self.arm.read()["gripper"], self.g_open, self.g_closed)

    def _target(self, name: str, offsets: bool) -> dict:
        target = {j: float(self.poses[name][j]) for j in ARM_JOINTS}
        if offsets:
            target["shoulder_pan"] += self.pan
            target["wrist_roll"] += self.roll
        return target

    def move(self, name: str, grip: float, offsets: bool = False, speed: float | None = None) -> None:
        """Move all arm joints together to a pose at `speed` (default self.speed). The claw
        target is sent straight away, never ramped (ramping loosens the grip and drops the object)."""
        self.log(f"move {name}")
        target = self._target(name, offsets)
        pos = self.arm.read()
        start = {j: pos[j] for j in ARM_JOINTS}
        duration = max(abs(target[j] - start[j]) for j in ARM_JOINTS) / (speed or self.speed)
        t0 = self.clock.now()
        while True:
            self._check_stop()
            elapsed = self.clock.now() - t0
            frac = 1.0 if duration <= 0 else min(1.0, elapsed / duration)
            self._send({j: start[j] + (target[j] - start[j]) * frac for j in ARM_JOINTS}, grip)
            if frac >= 1.0:
                pos = self.arm.read()
                late = [j for j in ARM_JOINTS if abs(pos[j] - target[j]) > TOLERANCE_DEG[j]]
                if not late:
                    return
                if elapsed > duration + STUCK_AFTER_S:
                    self._hold()
                    raise _Abort("stuck", f"{name}: " + ", ".join(
                        f"{j} at {pos[j]:.1f} (target {target[j]:.1f})" for j in late))
            self.clock.sleep(self.dt)

    def set_grip(self, grip: float) -> bool:
        """Move only the claw; wait until it stops (on an object or at its target).
        Returns True if it stopped, False if it was still moving at GRIP_TIMEOUT_S."""
        self.log(f"claw {'open' if grip == self.g_open else 'close'}")
        t0 = still_since = self.clock.now()
        last = self.arm.read()["gripper"]
        while True:
            self._check_stop()
            self._send(self.joint_cmd or {j: self.arm.read()[j] for j in ARM_JOINTS}, grip)
            self.clock.sleep(self.dt)
            now = self.clock.now()
            g = self.arm.read()["gripper"]
            if abs(g - last) > 0.2:
                still_since = now
                last = g
            if now - still_since >= GRIP_SETTLE_S:
                return True
            if now - t0 >= GRIP_TIMEOUT_S:
                return False

    def _closed_side(self, gripper: float) -> bool:
        """Within GRIP_MARGIN of the taught closed value (or past it)."""
        return (gripper - self.g_closed) * self.dir < GRIP_MARGIN

    def policy_grasp(self, policy) -> bool:
        """Let the policy drive, one step per frame at RATE_HZ, for up to POLICY_TIME_LIMIT_S.
        It hands over once it asks for a closed claw and the claw has stood still (moved at
        most GRIP_STILL_DEG per frame) for POLICY_HOLD_FRAMES frames in a row, short of closed
        (on something) or fully closed (on nothing). Then, or at the time limit, the claw is
        squeezed shut and waited on like a scripted grasp. Returns True only if the claw then
        stopped short of closed, i.e. holds something."""
        self.log("policy grasp")
        if hasattr(policy, "reset"):
            policy.reset()
        state = self.arm.read()
        # Start from the last command, not the reading: the joints settle a few degrees short
        # (soft servos, lag), and stepping from the reading would break the speed cap.
        cmd = dict(self.joint_cmd) if self.joint_cmd else {j: state[j] for j in ARM_JOINTS}
        last_grip = state["gripper"]
        max_step = self.speed * self.dt
        held = empty = 0
        t0 = next_tick = self.clock.now()
        while self.clock.now() - t0 < POLICY_TIME_LIMIT_S:
            self._check_stop()
            state = self.arm.read()
            try:
                action = policy.step(state)
                target = {k: float(action[k]) for k in (*ARM_JOINTS, "gripper")}
                # A NaN target would slip through the speed cap below as a full step every frame.
                bad = [k for k, v in target.items() if not math.isfinite(v)]
                if bad:
                    raise ValueError(f"policy returned a non-finite target for {', '.join(bad)}")
            except BaseException:   # camera stalled, inference failed or a bad target: hold, then report
                self._hold()
                raise
            for j in ARM_JOINTS:     # same speed cap as scripted moves
                cmd[j] += max(-max_step, min(max_step, target[j] - cmd[j]))
            grip = target["gripper"]
            self._send(cmd, grip)
            g = state["gripper"]
            still = abs(g - last_grip) <= GRIP_STILL_DEG
            last_grip = g
            wants_closed = (grip - self.g_closed) * self.dir <= GRIP_MARGIN
            on_object = wants_closed and still and claw_holding(g, self.g_open, self.g_closed)
            on_nothing = wants_closed and still and self._closed_side(g)
            held = held + 1 if on_object else 0
            empty = empty + 1 if on_nothing else 0
            if held >= POLICY_HOLD_FRAMES or empty >= POLICY_HOLD_FRAMES:
                self.log("policy closed the claw" + ("" if held else " on nothing"))
                break
            # Pace to RATE_HZ: sleep only what is left of this frame after the camera read,
            # inference and the serial round trip. Running late: start the next frame now.
            next_tick += self.dt
            delay = next_tick - self.clock.now()
            if delay > 0:
                self.clock.sleep(delay)
            else:
                next_tick = self.clock.now()
        else:
            self.log("policy time limit reached")
        stopped = self.set_grip(self.g_squeeze)
        return stopped and self.holding()

    def _light_target(self, hold_pos: float) -> float:
        """The light grip's claw target: LIGHT_SQUEEZE past hold_pos, never past g_squeeze."""
        target = hold_pos - self.dir * LIGHT_SQUEEZE
        return target if (target - self.g_squeeze) * self.dir >= 0 else self.g_squeeze

    def _reads_closed(self, gripper: float, hold_pos: float) -> bool:
        """On the closed side, or at least TAKE_CLOSE_UNITS closed from hold_pos."""
        return self._closed_side(gripper) or (hold_pos - gripper) * self.dir >= TAKE_CLOSE_UNITS

    def wait_for_take(self, timeout_s: float, hold_pos: float, grip: str = "squeeze") -> str:
        """Hold still at the present pose and read the claw at TAKE_POLL_HZ. hold_pos is where
        the claw read on arriving, holding. grip "squeeze" keeps squeezing; "light" commands
        _light_target(hold_pos) instead. Returns "taken" once the claw reads closed (on the
        closed side, or TAKE_CLOSE_UNITS closed from hold_pos) TAKE_READS times in a row (the
        person pulled the object out and the claw closed on nothing), or "dropped" if that
        closing began within TAKE_SETTLE_S of arriving (nobody had reached yet: it slipped out)
        or the claw reads open TAKE_READS times in a row. "timeout" if timeout_s passes first."""
        self.log(f"wait for take ({grip} grip)")
        target = self.g_squeeze if grip == "squeeze" else self._light_target(hold_pos)
        dt = 1.0 / TAKE_POLL_HZ
        closed = opened = 0
        closed_since = None
        t0 = self.clock.now()
        while True:
            now = self.clock.now()
            if now - t0 >= timeout_s:
                return "timeout"
            self._check_stop()
            # Re-send the claw target at every read: LeRobot caps each command's step
            # (max_relative_target), so one command alone would stop the claw short of closed.
            self._send(self.joint_cmd, target)
            g = self.arm.read()["gripper"]
            is_closed = self._reads_closed(g, hold_pos)
            if is_closed:
                closed_since = now if closed == 0 else closed_since
                closed += 1
            else:
                closed = 0
            opened = 0 if is_closed or claw_holding(g, self.g_open, self.g_closed) else opened + 1
            if closed >= TAKE_READS:
                if closed_since - t0 < TAKE_SETTLE_S:
                    self.log("claw closed before anyone could reach: the object slipped out")
                    return "dropped"
                self.log("taken")
                return "taken"
            if opened >= TAKE_READS:
                self.log("claw reads open: the object fell")
                return "dropped"
            self.clock.sleep(dt)

    # --- sequence ------------------------------------------------------------

    def _back_to_ready(self, speed: float | None = None) -> None:
        self.set_grip(self.g_open)
        self.move("ready", self.g_open, speed=speed)

    def carry(self, name: str, speed: float | None = None) -> None:
        """Move to a pose squeezing the object. If the claw no longer holds it there: open,
        go back to ready and report "dropped"."""
        self.move(name, self.g_squeeze, speed=speed)
        if not self.holding():
            self._back_to_ready(speed)
            raise _Abort("dropped", f"object lost on the way to {name}")

    def pick_up(self, policy, result: dict) -> None:
        """ready -> grasp (scripted or policy, with retries) -> straight up -> lift, holding."""
        self.move("ready", self.g_open)
        while True:
            result["attempts"] += 1
            if policy is not None:
                holding = self.policy_grasp(policy)
            else:
                self.move("above_object", self.g_open, offsets=True)
                self.move("at_object", self.g_open, offsets=True)
                self.set_grip(self.g_squeeze)
                holding = self.holding()
            if holding:
                break
            self.log("claw closed on nothing")
            self.set_grip(self.g_open)
            self.move("above_object", self.g_open, offsets=True)
            self.move("ready", self.g_open)
            if result["attempts"] >= ATTEMPTS:
                raise _Abort("missed", f"claw empty after {ATTEMPTS} attempts")

        self.move("above_object", self.g_squeeze, offsets=True)
        self.carry("lift")

    def run_tray(self, policy, result: dict) -> None:
        self.pick_up(policy, result)
        self.carry("above_tray")
        self.set_grip(self.g_open)
        self.move("lift", self.g_open)
        self.move("ready", self.g_open)
        result["detail"] = "released over the tray"

    def run_hand(self, policy, result: dict, timeout_s: float, on_state, grip: str = "squeeze") -> None:
        on_state("pick")
        self.pick_up(policy, result)
        near_person = min(self.speed, PRESENT_SPEED_DEG_S)
        on_state("present")
        self.carry("present", speed=near_person)
        hold_pos = self.arm.read()["gripper"]        # holding: carry() just checked
        on_state("wait_for_take")
        outcome = self.wait_for_take(timeout_s, hold_pos, grip)
        if outcome == "dropped":
            self._back_to_ready(near_person)
            raise _Abort("dropped", "object slipped out at the present pose before anyone could take it")
        if outcome == "taken":
            self.set_grip(self.g_open)
            self.move("ready", self.g_open, speed=near_person)
            result["detail"] = "taken at the present pose"
            return
        self.carry("lift", speed=near_person)       # squeezing again; "timeout" promises it still holds (plan 9.4)
        result["result"] = "timeout"
        result["detail"] = f"not taken within {timeout_s:g} s; holding at lift"


def close_cameras(cameras: dict) -> list[str]:
    """Close every camera, even if one fails. Returns a note per failure."""
    failures = []
    for name, cam in cameras.items():
        try:
            cam.close()
        except Exception as e:  # noqa: BLE001 - the others must still close
            failures.append(f"camera {name} close failed: {type(e).__name__}: {e}")
    return failures


def load_policy(path, cameras: dict | None = None, config_path: str | Path | None = None,
                capture_size: tuple[int, int] | None = None, blank: bool = False,
                device: str | None = None):
    """Load an ACT checkpoint folder (its `pretrained_model`) onto the Mac's GPU.

    Without `cameras`, opens the webcams the checkpoint was trained on, by their
    robot.yaml `cameras` index, the way training/record.py (task T1) records them:
    LeRobot's OpenCV camera at the checkpoint's image size and RATE_HZ fps. If a
    webcam can't deliver that size, pass the size the demos were captured at as
    `capture_size` (width, height); it must have the same aspect ratio, and frames
    are resized to the checkpoint's size. blank=True gives grey frames instead of
    webcams (dry runs). Returns the policy; its cameras are in policy.cameras (close
    them when done). If loading fails part-way, every camera it opened is closed."""
    from . import policy as policy_mod

    sizes = policy_mod.checkpoint_image_sizes(path)
    if capture_size is not None:
        for name, size in sizes.items():
            if not policy_mod.same_aspect(capture_size, size):
                raise ValueError(f"capture size {capture_size[0]}x{capture_size[1]} has a different aspect "
                                 f"ratio from the checkpoint's {name} images ({size[0]}x{size[1]}); "
                                 f"resizing would squash them")
    opened = {}
    try:
        if cameras is None:
            indices = load_camera_indices(config_path if config_path is not None else DEFAULT_CONFIG_PATH,
                                          list(sizes))
            for name, size in sizes.items():
                width, height = capture_size or size
                if blank:
                    opened[name] = policy_mod.BlankFrames(width, height)
                else:
                    opened[name] = policy_mod.OpenCVFrames(indices[name], width, height, RATE_HZ)
            cameras = opened
        return policy_mod.ActGrasp(path, cameras, device=device)
    except BaseException:
        close_cameras(opened)
        raise


def _pick(mode, sequence, arm, poses, pan_deg, roll_deg, policy, speed_deg_s, stop_event, log) -> dict:
    """Shared by pick_to_hand and pick_to_tray. Checks the arguments and the poses `mode`
    needs before anything moves, then runs sequence(picker, policy, result).
    policy may be a checkpoint folder: its webcams are opened here and closed at the end."""
    if not (_is_number(speed_deg_s) and 0 < speed_deg_s <= MAX_SPEED_DEG_S):
        raise ValueError(f"speed_deg_s must be in (0, {MAX_SPEED_DEG_S:g}] deg/s (plan section 17), "
                         f"got {speed_deg_s!r}")
    for name, value in (("pan_deg", pan_deg), ("roll_deg", roll_deg)):
        if not _is_number(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    poses_mod.validate(poses, mode)
    opened = isinstance(policy, (str, Path))
    if opened:
        policy = load_policy(policy)    # closes its own cameras if it fails
    try:
        picker = _Picker(arm, poses, pan_deg, roll_deg, speed_deg_s, stop_event, log or (lambda msg: None))
        result = {"result": "taken" if mode == "hand" else "in_tray",
                  "grasp": "policy" if policy is not None else "scripted",
                  "attempts": 0, "seconds": 0.0, "detail": ""}
        t0 = picker.clock.now()
        try:
            sequence(picker, policy, result)
        except _Abort as abort:
            result["result"] = abort.result
            result["detail"] = abort.detail
        result["seconds"] = round(picker.clock.now() - t0, 2)
    finally:
        notes = close_cameras(policy.cameras) if opened else []
    if notes:
        result["detail"] = "; ".join(filter(None, [result["detail"], *notes]))
    return result


def pick_to_hand(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45,
                 handoff_timeout_s=60, stop_event: threading.Event | None = None, on_state=None,
                 log=None, handoff_grip="squeeze") -> dict:
    """Pick the object at the parking spot, present it and wait for the person to take it.
    See the module docstring. Needs the `present` pose (not `above_tray`).
    stop_event: set it from another thread to stop; the arm holds where it is and keeps its
    claw target (squeezing, or the light hold while waiting).
    on_state(state) is called with "pick", "present" and "wait_for_take".
    handoff_grip: "squeeze" or "light", the grip while waiting at `present` (D9).
    policy may be a checkpoint folder: its webcams are opened here and closed at the end."""
    if not (_is_number(handoff_timeout_s) and handoff_timeout_s > 0):
        raise ValueError(f"handoff_timeout_s must be a positive number of seconds, got {handoff_timeout_s!r}")
    if not (isinstance(handoff_grip, str) and handoff_grip in HANDOFF_GRIPS):
        raise ValueError(f"handoff_grip must be 'squeeze' or 'light', got {handoff_grip!r}")
    on_state = on_state or (lambda state: None)
    sequence = functools.partial(_Picker.run_hand, timeout_s=handoff_timeout_s, on_state=on_state,
                                 grip=handoff_grip)
    return _pick("hand", sequence, arm, poses, pan_deg, roll_deg, policy, speed_deg_s, stop_event, log)


def pick_to_tray(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45,
                 stop_event: threading.Event | None = None, log=None) -> dict:
    """Pick the object at the parking spot and drop it in the tray (stretch). See the module
    docstring. Needs the `above_tray` pose (not `present`).
    stop_event: set it from another thread to stop; the arm holds where it is.
    policy may be a checkpoint folder: its webcams are opened here and closed at the end."""
    return _pick("tray", _Picker.run_tray, arm, poses, pan_deg, roll_deg, policy, speed_deg_s, stop_event, log)


# --- configuration -----------------------------------------------------------

class ConfigError(ValueError):
    pass


def _load_config(path: Path) -> dict:
    """Raw robot.yaml contents, {} if the file is missing. Prefer load_arm_config(),
    which never falls back to defaults."""
    if not path.exists():
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_arm_config(path: str | Path = DEFAULT_CONFIG_PATH,
                    require: tuple[str, ...] = ("follower_port", "follower_id")) -> dict:
    """The `arm:` section of robot.yaml, checked. The safety limits max_relative_target_deg
    and move_speed_deg_s must be positive numbers (no defaults), move_speed_deg_s no more
    than MAX_SPEED_DEG_S, and every key in `require` a non-empty string. Raises ConfigError."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found; robot.yaml comes from task S6 (plan section 9.1)")
    cfg = _load_config(path)
    arm = cfg.get("arm") if isinstance(cfg, dict) else None
    if not isinstance(arm, dict):
        raise ConfigError(f"{path}: the 'arm:' section is missing or empty")
    for key in require:
        value = arm.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{path}: arm.{key} is missing or empty")
    for key in ("max_relative_target_deg", "move_speed_deg_s"):
        value = arm.get(key)
        if not _is_number(value) or value <= 0:
            raise ConfigError(f"{path}: arm.{key} must be a positive number, got {value!r}")
    if arm["move_speed_deg_s"] > MAX_SPEED_DEG_S:
        raise ConfigError(f"{path}: arm.move_speed_deg_s is {arm['move_speed_deg_s']:g}, above the "
                          f"{MAX_SPEED_DEG_S:g} deg/s arm limit (plan section 17)")
    return arm


def _load_handoff(path: Path, key: str):
    """robot.yaml handoff.<key>, unchecked. Raises ConfigError if the file or section is missing."""
    if not path.exists():
        raise ConfigError(f"{path} not found; robot.yaml comes from task S6 (plan section 9.1)")
    cfg = _load_config(path)
    handoff = cfg.get("handoff") if isinstance(cfg, dict) else None
    if not isinstance(handoff, dict):
        raise ConfigError(f"{path}: the 'handoff:' section is missing or empty (it needs handoff.{key})")
    return handoff.get(key)


def load_handoff_timeout(path: str | Path = DEFAULT_CONFIG_PATH) -> float:
    """robot.yaml handoff.timeout_s: seconds holding at the present pose before giving up.
    It must be a positive number (no default). Raises ConfigError."""
    path = Path(path)
    value = _load_handoff(path, "timeout_s")
    if not _is_number(value) or value <= 0:
        raise ConfigError(f"{path}: handoff.timeout_s must be a positive number of seconds, got {value!r}")
    return float(value)


def load_handoff_grip(path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """robot.yaml handoff.grip: the claw grip while waiting at the present pose (D9), exactly
    "squeeze" or "light" (no default). Raises ConfigError."""
    path = Path(path)
    value = _load_handoff(path, "grip")
    if not (isinstance(value, str) and value in HANDOFF_GRIPS):
        raise ConfigError(f"{path}: handoff.grip must be 'squeeze' or 'light', got {value!r}")
    return value


def load_camera_indices(path: str | Path, names: list[str]) -> dict:
    """{name: OpenCV index or path} from robot.yaml `cameras` for each camera in names.
    Raises ConfigError if the file, the section or any of the cameras is missing."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found; robot.yaml comes from task S6 (plan section 9.1)")
    cfg = _load_config(path)
    cams = cfg.get("cameras") if isinstance(cfg, dict) else None
    if not isinstance(cams, dict):
        raise ConfigError(f"{path}: the 'cameras:' section is missing or empty")
    missing = [n for n in names if n not in cams]
    if missing:
        raise ConfigError(f"{path}: cameras has no index for {missing} (the checkpoint needs {names})")
    for n in names:
        value = cams[n]
        if isinstance(value, bool) or not (isinstance(value, int) or (isinstance(value, str) and value.strip())):
            raise ConfigError(f"{path}: cameras.{n} must be an OpenCV index or device path, got {value!r}")
    return {n: cams[n] for n in names}


def resolve_speed(requested: float | None, arm_cfg: dict) -> float:
    """The speed to run at: robot.yaml arm.move_speed_deg_s unless a lower one is requested.
    A requested speed above that ceiling is refused (ValueError), never clipped."""
    ceiling = float(arm_cfg["move_speed_deg_s"])
    if requested is None:
        return ceiling
    if not _is_number(requested) or requested <= 0:
        raise ValueError(f"speed must be a positive number of deg/s, got {requested!r}")
    if requested > ceiling:
        raise ValueError(f"speed {requested:g} deg/s is above the robot.yaml ceiling "
                         f"arm.move_speed_deg_s = {ceiling:g}; refusing to run")
    return float(requested)


# --- command line ------------------------------------------------------------

class _ArgError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Bad arguments end in a RESULT line and exit 1 (argparse's own exit 2 means 'missed')."""

    def error(self, message):
        self.print_usage(sys.stderr)
        text = f"{self.prog}: error: {message}"
        print(text, file=sys.stderr)
        raise _ArgError(text)


def _size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 640x360, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"width and height must be positive, got {text!r}")
    return width, height


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m arm.pick")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="pick the object at the parking spot and hand it to the person "
                                     "(or, --to tray, drop it in the tray)")
    run.add_argument("--to", choices=("hand", "tray"), default="hand",
                     help="hand: present it and wait for the take (default); tray: stretch tray mode")
    run.add_argument("--policy", help="ACT checkpoint folder (pretrained_model)")
    run.add_argument("--pan", type=float, default=0.0, help="shoulder_pan offset, deg")
    run.add_argument("--roll", type=float, default=0.0, help="wrist_roll offset, deg")
    run.add_argument("--speed", type=float,
                     help="deg/s, at most robot.yaml arm.move_speed_deg_s (the default); "
                          "first real runs: 25")
    run.add_argument("--handoff-timeout", type=float, metavar="S",
                     help="hand mode: seconds to hold at the present pose before giving up "
                          "(default: robot.yaml handoff.timeout_s)")
    run.add_argument("--handoff-grip", choices=HANDOFF_GRIPS,
                     help="hand mode: claw grip while waiting at the present pose, squeeze (firm) or "
                          "light (default: robot.yaml handoff.grip)")
    run.add_argument("--dry-run", action="store_true", help="simulated arm, no hardware")
    run.add_argument("--camera-size", type=_size, metavar="WxH",
                     help="with --policy: webcam capture size, if a webcam can't deliver the checkpoint's "
                          "image size directly (use the size the demos were recorded at; same aspect ratio)")
    run.add_argument("--poses", type=Path, default=poses_mod.DEFAULT_POSES_PATH)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument("--sim-miss", type=int, default=0, help="dry run: first N grasps miss")
    run.add_argument("--sim-slip", action="store_true", help="dry run: object slips out while lifting")
    run.add_argument("--sim-stall", metavar="JOINT", choices=ARM_JOINTS,
                     help="dry run: this joint doesn't move (one of %(choices)s)")
    run.add_argument("--sim-take-after", type=float, default=2.0, metavar="S",
                     help="dry run: the person takes the object S s after it is presented (default 2; "
                          "under TAKE_SETTLE_S = 1 s it reads as a slip)")
    run.add_argument("--sim-never-take", action="store_true", help="dry run: nobody takes the object")
    return parser


def _error_result(detail: str, grasp: str) -> dict:
    return {"result": "error", "grasp": grasp, "attempts": 0, "seconds": 0.0, "detail": detail}


def _finish(result: dict) -> int:
    print("RESULT " + json.dumps(result), flush=True)
    return EXIT_CODES[result["result"]]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        for flag, value in (("--handoff-timeout", args.handoff_timeout), ("--handoff-grip", args.handoff_grip)):
            if args.to == "tray" and value is not None:
                parser.error(f"{flag} only applies to --to hand")
    except _ArgError as e:
        policy_given = any(a == "--policy" or a.startswith("--policy=") for a in argv)
        return _finish(_error_result(str(e), "policy" if policy_given else "scripted"))

    grasp = "policy" if args.policy else "scripted"
    stop_event = threading.Event()
    log = lambda msg: print(msg, flush=True)
    old_handlers = {}
    arm = None
    cameras = {}
    result = None
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):   # Ctrl-C or kill: stop, arm holds
                old_handlers[sig] = signal.signal(sig, lambda *_: stop_event.set())
        cfg = load_arm_config(args.config)
        speed = resolve_speed(args.speed, cfg)
        for flag, value in (("--pan", args.pan), ("--roll", args.roll)):   # before any hardware opens
            if not _is_number(value):
                raise ValueError(f"{flag} must be a finite number, got {value!r}")
        timeout = grip = None
        if args.to == "hand":
            timeout = args.handoff_timeout
            if timeout is None:
                timeout = load_handoff_timeout(args.config)
            elif not (_is_number(timeout) and timeout > 0):
                raise ValueError(f"--handoff-timeout must be a positive number of seconds, got {timeout!r}")
            grip = args.handoff_grip if args.handoff_grip is not None else load_handoff_grip(args.config)
        if args.dry_run:
            from .sim import DEFAULT_POSES, SimArm
            poses = poses_mod.load(args.poses, args.to) if args.poses.exists() else DEFAULT_POSES
        else:
            poses = poses_mod.load(args.poses, args.to)     # a pose this mode needs is missing: stop here
        policy = None
        if args.policy:     # before the arm opens; a dry run uses grey frames instead of webcams
            policy = load_policy(args.policy, config_path=args.config, capture_size=args.camera_size,
                                 blank=args.dry_run)
            cameras = policy.cameras
        if args.dry_run:
            arm = SimArm(poses, miss_first=args.sim_miss, slip_on_lift=args.sim_slip,
                         stall_joint=args.sim_stall, take_after_s=args.sim_take_after,
                         never_take=args.sim_never_take)
        else:
            from .real import RealArm
            arm = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                          max_relative_target=cfg["max_relative_target_deg"])
        if args.to == "hand":
            result = pick_to_hand(arm, poses, pan_deg=args.pan, roll_deg=args.roll, policy=policy,
                                  speed_deg_s=speed, handoff_timeout_s=timeout, stop_event=stop_event,
                                  on_state=lambda state: log(f"state {state}"), log=log, handoff_grip=grip)
        else:
            result = pick_to_tray(arm, poses, pan_deg=args.pan, roll_deg=args.roll, policy=policy,
                                  speed_deg_s=speed, stop_event=stop_event, log=log)
    except KeyboardInterrupt:     # only before the STOP handlers are in place
        result = {**_error_result("interrupted", grasp), "result": "stopped"}
    except Exception as e:  # noqa: BLE001 - any failure must still end in a RESULT line
        result = _error_result(f"{type(e).__name__}: {e}", grasp)
    finally:
        for note in close_cameras(cameras):
            if result is not None:
                result["detail"] = f"{result['detail']}; {note}" if result["detail"] else note
        if arm is not None:
            try:
                arm.close()
            except Exception as e:  # noqa: BLE001
                if result is not None:
                    note = f"close failed: {type(e).__name__}: {e}"
                    result["detail"] = f"{result['detail']}; {note}" if result["detail"] else note
        for sig, handler in old_handlers.items():
            if handler is not None:
                signal.signal(sig, handler)
    return _finish(result)


if __name__ == "__main__":
    sys.exit(main())
