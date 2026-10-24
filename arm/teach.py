"""Teach the pick poses with the leader arm (task A3, plan 10.1).

    python -m arm.teach [--out config/poses.json] [--speed DEG_S] [--dry-run]

The follower copies the leader, speed-capped. The person moves the leader and
presses Enter at each step; the follower's position is recorded (it's what
pick.py will command). Saves config/poses.json, then reloads it to check. An
existing poses.json is backed up to data/poses_backup/ first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

from . import poses as poses_mod
from .pick import RATE_HZ, RealClock, _load_config
from .poses import ARM_JOINTS, JOINTS, POSE_NAMES, REPO_ROOT

STEPS = [
    ("ready", "the start pose: arm up and out of the front camera's view"),
    ("above_object", "claw open, a few cm straight above the object on the parking spot"),
    ("at_object", "claw open, fingertips just above the floor around the object"),
    ("lift", "the carry pose: object raised clear of the deck"),
    ("above_tray", "claw over the tray at its lowest position"),
    ("gripper_open", "open the claw fully"),
    ("gripper_closed", "close the claw fully, with nothing in it"),
]


def enter_waiter(prompt: str) -> threading.Event:
    """Return an event that is set when the person presses Enter."""
    pressed = threading.Event()

    def wait():
        input(prompt)
        pressed.set()

    threading.Thread(target=wait, daemon=True).start()
    return pressed


def teach(leader, follower, speed_deg_s: float = 45, wait_for=enter_waiter, log=print) -> dict:
    """Mirror leader -> follower; record the follower at each step. Returns validated poses."""
    clock = getattr(follower, "clock", None) or RealClock()
    dt = 1.0 / RATE_HZ
    max_step = speed_deg_s * dt
    cmd = follower.read()
    recorded = {}
    try:
        for name, hint in STEPS:
            pressed = wait_for(f"[{name}] {hint}. Press Enter here when the follower is there: ")
            while not pressed.is_set():
                target = leader.read()
                for j in ARM_JOINTS:
                    cmd[j] += max(-max_step, min(max_step, target[j] - cmd[j]))
                cmd["gripper"] = target["gripper"]
                follower.command(cmd)
                clock.sleep(dt)
            recorded[name] = follower.read()
            log(f"  {name}: " + ", ".join(f"{j} {recorded[name][j]:.1f}" for j in JOINTS))
    finally:
        follower.command(follower.read())       # hold where it is
    poses = {name: recorded[name] for name in POSE_NAMES}
    poses["gripper_open"] = recorded["gripper_open"]["gripper"]
    poses["gripper_closed"] = recorded["gripper_closed"]["gripper"]
    return poses_mod.validate(poses)


def save_with_backup(poses: dict, out: Path) -> Path | None:
    backup = None
    if out.exists():
        backup = REPO_ROOT / "data" / "poses_backup" / f"poses-{time.strftime('%Y%m%d-%H%M%S')}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, backup)
    poses_mod.save(poses, out)
    poses_mod.load(out)                         # reload check
    return backup


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arm.teach")
    parser.add_argument("--out", type=Path, default=poses_mod.DEFAULT_POSES_PATH)
    parser.add_argument("--speed", type=float, help="follower speed cap, deg/s (default: robot.yaml, else 25)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "robot.yaml")
    parser.add_argument("--dry-run", action="store_true", help="simulated arms; follower copies a still leader")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config).get("arm", {})
    speed = args.speed or cfg.get("move_speed_deg_s") or 25
    if args.dry_run:
        from .sim import SimArm, SimLeader
        follower, leader = SimArm(), SimLeader()
        follower.clock.realtime = True
    else:
        from .real import RealArm, RealLeader
        follower = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                           max_relative_target=cfg.get("max_relative_target_deg", 10))
        leader = RealLeader(port=cfg["leader_port"], robot_id=cfg["leader_id"])
    try:
        poses = teach(leader, follower, speed_deg_s=speed)
    except KeyboardInterrupt:
        print("\nstopped; nothing saved")
        return 1
    except poses_mod.PosesError as e:
        print(f"not saved: {e}")
        return 1
    finally:
        follower.close()
        leader.close()
    backup = save_with_backup(poses, args.out)
    print(f"saved and reloaded {args.out}" + (f" (old file backed up to {backup})" if backup else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
