"""Pick sequence: floor -> tray (plan sections 9.4 and 10.1).

    result = pick_to_tray(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45)

    python -m arm.pick run [--policy DIR] [--pan DEG] [--roll DEG] [--speed DEG_S] [--dry-run]

The last printed line is `RESULT {json}`. Exit codes: 0 in_tray, 1 error or stopped,
2 missed, 3 dropped, 4 stuck.

`arm` is anything with read() -> {joint: value}, command({joint: value}) and close():
arm.sim.SimArm or arm.real.RealArm. If it has a `clock` (the sim does), timing runs
on it; otherwise on the wall clock.

`policy` (task T3) is an object with step(state) -> {joint: target} for all six
joints, and optionally reset(). It drives from `ready` until it asks for a closed
claw and the claw has been stopped short for 8 frames; the scripted lift takes over.

Speed: config/robot.yaml `arm.move_speed_deg_s` is the ceiling (plan 9.1, 17). The
command line runs at that speed by default and refuses a larger --speed; first real
runs use --speed 25. pick_to_tray() itself also refuses anything above the plan's
45 deg/s arm limit (MAX_SPEED_DEG_S), for callers that skip load_arm_config().
"""

from __future__ import annotations

import argparse
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
POLICY_TIME_LIMIT_S = 20.0
MAX_SPEED_DEG_S = 45.0          # plan section 17 arm limit; robot.yaml move_speed_deg_s is the working ceiling
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"

EXIT_CODES = {"in_tray": 0, "stopped": 1, "error": 1, "missed": 2, "dropped": 3, "stuck": 4}


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

    def move(self, name: str, grip: float, offsets: bool = False) -> None:
        """Move all arm joints together to a pose at self.speed. The claw target is sent
        straight away, never ramped (ramping loosens the grip and drops the object)."""
        self.log(f"move {name}")
        target = self._target(name, offsets)
        pos = self.arm.read()
        start = {j: pos[j] for j in ARM_JOINTS}
        duration = max(abs(target[j] - start[j]) for j in ARM_JOINTS) / self.speed
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

    def set_grip(self, grip: float) -> None:
        """Move only the claw; wait until it stops (on an object or at its target)."""
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
            if now - still_since >= GRIP_SETTLE_S or now - t0 >= GRIP_TIMEOUT_S:
                return

    def policy_grasp(self, policy) -> None:
        """Let the policy drive until it closes on something for POLICY_HOLD_FRAMES frames."""
        self.log("policy grasp")
        if hasattr(policy, "reset"):
            policy.reset()
        state = self.arm.read()
        cmd = {j: state[j] for j in ARM_JOINTS}
        max_step = self.speed * self.dt
        held = 0
        t0 = self.clock.now()
        while self.clock.now() - t0 < POLICY_TIME_LIMIT_S:
            self._check_stop()
            state = self.arm.read()
            action = policy.step(state)
            for j in ARM_JOINTS:     # same speed cap as scripted moves
                cmd[j] += max(-max_step, min(max_step, float(action[j]) - cmd[j]))
            grip = float(action["gripper"])
            self._send(cmd, grip)
            wants_closed = (grip - self.g_closed) * self.dir <= GRIP_MARGIN
            held = held + 1 if wants_closed and claw_holding(state["gripper"], self.g_open, self.g_closed) else 0
            if held >= POLICY_HOLD_FRAMES:
                self._send(cmd, self.g_squeeze)
                return
            self.clock.sleep(self.dt)
        self.log("policy time limit reached")

    # --- sequence ------------------------------------------------------------

    def _back_to_ready(self) -> None:
        self.set_grip(self.g_open)
        self.move("ready", self.g_open)

    def run(self, policy, result: dict) -> None:
        self.move("ready", self.g_open)
        while True:
            result["attempts"] += 1
            if policy is not None:
                self.policy_grasp(policy)
            else:
                self.move("above_object", self.g_open, offsets=True)
                self.move("at_object", self.g_open, offsets=True)
                self.set_grip(self.g_squeeze)
            if self.holding():
                break
            self.log("claw closed on nothing")
            self.set_grip(self.g_open)
            self.move("above_object", self.g_open, offsets=True)
            self.move("ready", self.g_open)
            if result["attempts"] >= ATTEMPTS:
                raise _Abort("missed", f"claw empty after {ATTEMPTS} attempts")

        self.move("above_object", self.g_squeeze, offsets=True)
        for name in ("lift", "above_tray"):
            self.move(name, self.g_squeeze)
            if not self.holding():
                self._back_to_ready()
                raise _Abort("dropped", f"object lost on the way to {name}")
        self.set_grip(self.g_open)
        self.move("lift", self.g_open)
        self.move("ready", self.g_open)
        result["detail"] = "released over the tray"


def load_policy(path):
    raise NotImplementedError("loading an ACT checkpoint is task T3")


def pick_to_tray(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45,
                 stop_event: threading.Event | None = None, log=None) -> dict:
    """Pick the object at the parking spot and drop it in the tray. See the module docstring.
    stop_event: set it from another thread to stop; the arm holds where it is."""
    if not (_is_number(speed_deg_s) and 0 < speed_deg_s <= MAX_SPEED_DEG_S):
        raise ValueError(f"speed_deg_s must be in (0, {MAX_SPEED_DEG_S:g}] deg/s (plan section 17), "
                         f"got {speed_deg_s!r}")
    for name, value in (("pan_deg", pan_deg), ("roll_deg", roll_deg)):
        if not _is_number(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    poses_mod.validate(poses)
    if isinstance(policy, (str, Path)):
        policy = load_policy(policy)
    picker = _Picker(arm, poses, pan_deg, roll_deg, speed_deg_s, stop_event, log or (lambda msg: None))
    result = {"result": "in_tray", "grasp": "policy" if policy is not None else "scripted",
              "attempts": 0, "seconds": 0.0, "detail": ""}
    t0 = picker.clock.now()
    try:
        picker.run(policy, result)
    except _Abort as abort:
        result["result"] = abort.result
        result["detail"] = abort.detail
    result["seconds"] = round(picker.clock.now() - t0, 2)
    return result


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


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m arm.pick")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="pick the object at the parking spot into the tray")
    run.add_argument("--policy", help="ACT checkpoint folder (pretrained_model)")
    run.add_argument("--pan", type=float, default=0.0, help="shoulder_pan offset, deg")
    run.add_argument("--roll", type=float, default=0.0, help="wrist_roll offset, deg")
    run.add_argument("--speed", type=float,
                     help="deg/s, at most robot.yaml arm.move_speed_deg_s (the default); "
                          "first real runs: 25")
    run.add_argument("--dry-run", action="store_true", help="simulated arm, no hardware")
    run.add_argument("--poses", type=Path, default=poses_mod.DEFAULT_POSES_PATH)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument("--sim-miss", type=int, default=0, help="dry run: first N grasps miss")
    run.add_argument("--sim-slip", action="store_true", help="dry run: object slips out while lifting")
    run.add_argument("--sim-stall", metavar="JOINT", choices=ARM_JOINTS,
                     help="dry run: this joint doesn't move (one of %(choices)s)")
    return parser


def _error_result(detail: str, grasp: str) -> dict:
    return {"result": "error", "grasp": grasp, "attempts": 0, "seconds": 0.0, "detail": detail}


def _finish(result: dict) -> int:
    print("RESULT " + json.dumps(result), flush=True)
    return EXIT_CODES[result["result"]]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        args = _build_parser().parse_args(argv)
    except _ArgError as e:
        policy_given = any(a == "--policy" or a.startswith("--policy=") for a in argv)
        return _finish(_error_result(str(e), "policy" if policy_given else "scripted"))

    grasp = "policy" if args.policy else "scripted"
    stop_event = threading.Event()
    log = lambda msg: print(msg, flush=True)
    old_handlers = {}
    arm = None
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
        if args.dry_run:
            from .sim import DEFAULT_POSES, SimArm
            poses = poses_mod.load(args.poses) if args.poses.exists() else DEFAULT_POSES
        else:
            poses = poses_mod.load(args.poses)
        policy = load_policy(args.policy) if args.policy else None   # before any hardware opens
        if args.dry_run:
            arm = SimArm(poses, miss_first=args.sim_miss, slip_on_lift=args.sim_slip,
                         stall_joint=args.sim_stall)
        else:
            from .real import RealArm
            arm = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                          max_relative_target=cfg["max_relative_target_deg"])
        result = pick_to_tray(arm, poses, pan_deg=args.pan, roll_deg=args.roll, policy=policy,
                              speed_deg_s=speed, stop_event=stop_event, log=log)
    except KeyboardInterrupt:     # only before the STOP handlers are in place
        result = {**_error_result("interrupted", grasp), "result": "stopped"}
    except Exception as e:  # noqa: BLE001 - any failure must still end in a RESULT line
        result = _error_result(f"{type(e).__name__}: {e}", grasp)
    finally:
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
