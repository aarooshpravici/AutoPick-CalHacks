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
"""

from __future__ import annotations

import argparse
import json
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
MAX_SPEED_DEG_S = 90.0

EXIT_CODES = {"in_tray": 0, "stopped": 1, "error": 1, "missed": 2, "dropped": 3, "stuck": 4}


class RealClock:
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


class _Abort(Exception):
    def __init__(self, result: str, detail: str):
        super().__init__(detail)
        self.result = result
        self.detail = detail


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


def load_policy(path, cameras: dict | None = None, config_path: Path | None = None):
    """Load an ACT checkpoint folder. Without `cameras`, opens the webcams named in
    robot.yaml `cameras` that the checkpoint needs. Returns the policy; its cameras
    are in policy.cameras (close them when done)."""
    from .policy import ActGrasp, OpenCVFrames

    if cameras is None:
        indices = _load_config(config_path or REPO_ROOT / "config" / "robot.yaml").get("cameras", {})
        with open(Path(path) / "config.json") as f:
            needed = [k.removeprefix("observation.images.") for k in json.load(f)["input_features"]
                      if k.startswith("observation.images.")]
        missing = [n for n in needed if n not in indices]
        if missing:
            raise ValueError(f"robot.yaml cameras has no index for {missing}")
        cameras = {n: OpenCVFrames(indices[n]) for n in needed}
    return ActGrasp(path, cameras)


def pick_to_tray(arm, poses, pan_deg=0.0, roll_deg=0.0, policy=None, speed_deg_s=45,
                 stop_event: threading.Event | None = None, log=None) -> dict:
    """Pick the object at the parking spot and drop it in the tray. See the module docstring.
    stop_event: set it from another thread to stop; the arm holds where it is."""
    if not 0 < speed_deg_s <= MAX_SPEED_DEG_S:
        raise ValueError(f"speed_deg_s must be in (0, {MAX_SPEED_DEG_S}]")
    poses_mod.validate(poses)
    opened = isinstance(policy, (str, Path))
    if opened:
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
    finally:
        if opened:
            for cam in policy.cameras.values():
                cam.close()
    result["seconds"] = round(picker.clock.now() - t0, 2)
    return result


# --- command line ------------------------------------------------------------

def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arm.pick")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="pick the object at the parking spot into the tray")
    run.add_argument("--policy", help="ACT checkpoint folder (pretrained_model)")
    run.add_argument("--pan", type=float, default=0.0, help="shoulder_pan offset, deg")
    run.add_argument("--roll", type=float, default=0.0, help="wrist_roll offset, deg")
    run.add_argument("--speed", type=float, help="deg/s (default: robot.yaml arm.move_speed_deg_s, else 25)")
    run.add_argument("--dry-run", action="store_true", help="simulated arm, no hardware")
    run.add_argument("--poses", type=Path, default=poses_mod.DEFAULT_POSES_PATH)
    run.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "robot.yaml")
    run.add_argument("--sim-miss", type=int, default=0, help="dry run: first N grasps miss")
    run.add_argument("--sim-slip", action="store_true", help="dry run: object slips out while lifting")
    run.add_argument("--sim-stall", metavar="JOINT", help="dry run: this joint doesn't move")
    args = parser.parse_args(argv)

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    log = lambda msg: print(msg, flush=True)
    arm = None
    opened_cameras = {}
    try:
        cfg = _load_config(args.config).get("arm", {})
        speed = args.speed or cfg.get("move_speed_deg_s") or 25
        policy = args.policy
        if args.dry_run:
            from .sim import DEFAULT_POSES, SimArm
            poses = poses_mod.load(args.poses) if args.poses.exists() else DEFAULT_POSES
            arm = SimArm(poses, miss_first=args.sim_miss, slip_on_lift=args.sim_slip,
                         stall_joint=args.sim_stall)
            if policy:      # real checkpoint, grey camera frames
                from .policy import BlankFrames
                needed = json.loads((Path(policy) / "config.json").read_text())["input_features"]
                policy = load_policy(policy, cameras={k.removeprefix("observation.images."): BlankFrames()
                                                      for k in needed if k.startswith("observation.images.")})
        else:
            from .real import RealArm
            poses = poses_mod.load(args.poses)
            arm = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                          max_relative_target=cfg.get("max_relative_target_deg", 10))
            if policy:
                policy = load_policy(policy, config_path=args.config)
                opened_cameras = policy.cameras
        result = pick_to_tray(arm, poses, pan_deg=args.pan, roll_deg=args.roll, policy=policy,
                              speed_deg_s=speed, stop_event=stop_event, log=log)
    except Exception as e:  # noqa: BLE001 - any failure must still end in a RESULT line
        result = {"result": "error", "grasp": "policy" if args.policy else "scripted",
                  "attempts": 0, "seconds": 0.0, "detail": f"{type(e).__name__}: {e}"}
    finally:
        for cam in opened_cameras.values():
            cam.close()
        if arm is not None:
            arm.close()
    print("RESULT " + json.dumps(result), flush=True)
    return EXIT_CODES[result["result"]]


if __name__ == "__main__":
    sys.exit(main())
