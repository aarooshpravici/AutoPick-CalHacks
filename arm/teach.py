"""Teach the pick poses with the leader arm (task A3, plan 10.1).

    python -m arm.teach [--out PATH] [--speed DEG_S] [--config PATH] [--with-tray] [--dry-run]

The follower copies the leader, speed-capped. The person moves the leader and
presses Enter at each step; the follower's position is recorded (it's what
pick.py will command). Saves config/poses.json, then reloads it to check. An
existing file is backed up to data/poses_backup/ first.

The steps are the hand-off poses (ready, above_object, at_object, lift, present),
then the claw open and closed. --with-tray also teaches above_tray, after present,
for the stretch tray mode.

Speed: config/robot.yaml `arm.move_speed_deg_s` is the ceiling, as for arm.pick.
The follower copies the leader at that speed by default; a larger --speed is
refused. First real runs use --speed 25.

--dry-run uses no hardware. A scripted person moves a simulated leader through
sim.DEFAULT_POSES and "presses Enter" once the simulated follower is there. It
saves to a new temporary folder unless --out is given, and never writes
config/poses.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import poses as poses_mod
from .pick import (DEFAULT_CONFIG_PATH, MAX_SPEED_DEG_S, RATE_HZ, RealClock, _is_number,
                   load_arm_config, resolve_speed)
from .poses import ARM_JOINTS, JOINTS, POSE_NAMES, REPO_ROOT

STEPS = [                   # hand mode; teach_steps(with_tray=True) adds TRAY_STEP after present
    ("ready", "the start pose: arm up and out of the front camera's view"),
    ("above_object", "claw open, a few cm straight above the object on the parking spot"),
    ("at_object", "claw open, fingertips just above the floor around the object"),
    ("lift", "the carry pose: object raised clear of the deck"),
    ("present", "the hand-off pose: claw raised high and pointing forward, in easy reach"),
    ("gripper_open", "open the claw fully"),
    ("gripper_closed", "close the claw fully, with nothing in it"),
]
TRAY_STEP = ("above_tray", "claw over the tray at its lowest position (stretch tray mode)")
CONFIG_KEYS = ("follower_port", "follower_id", "leader_port", "leader_id")
SAME_POSE_DEG = 2.0         # a pose this close to the previous one on every arm joint wasn't moved to
DRY_RUN_LIMIT_S = 600.0     # simulated seconds the scripted person waits for the follower per step


class TeachError(RuntimeError):
    pass


# --- waiting for the person --------------------------------------------------

def _flush_stdin() -> None:
    """Drop keys typed ahead, so an extra Enter can't record the next pose unmoved."""
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - no terminal to flush (piped, Windows, closed stdin)
        pass


class _EnterPress:
    """is_set() turns true when the person presses Enter (read on a background thread)."""

    def __init__(self, prompt: str):
        self._done = threading.Event()
        self._eof = False
        threading.Thread(target=self._wait, args=(prompt,), daemon=True).start()

    def _wait(self, prompt: str) -> None:
        try:
            input(prompt)
        except EOFError:
            self._eof = True
        self._done.set()

    def is_set(self) -> bool:
        if self._done.is_set() and self._eof:
            raise TeachError("input closed before Enter was pressed")
        return self._done.is_set()


def enter_waiter(step: str, prompt: str) -> _EnterPress:
    """Flush typed-ahead input, show the prompt, and return a press to poll with is_set()."""
    _flush_stdin()
    return _EnterPress(prompt)


def teach_steps(with_tray: bool = False) -> list:
    """The (name, hint) steps in order: STEPS, plus TRAY_STEP right after present."""
    if not with_tray:
        return list(STEPS)
    at = [name for name, _ in STEPS].index("present") + 1
    return [*STEPS[:at], TRAY_STEP, *STEPS[at:]]


def dry_run_targets(poses: dict, with_tray: bool = False) -> dict:
    """Where the scripted person puts the leader at each step: the poses, then the claw open
    and closed at the last of them (present, or above_tray with the tray)."""
    names = [name for name, _ in teach_steps(with_tray) if name in POSE_NAMES]
    targets = {name: dict(poses[name]) for name in names}
    targets["gripper_open"] = {**poses[names[-1]], "gripper": poses["gripper_open"]}
    targets["gripper_closed"] = {**poses[names[-1]], "gripper": poses["gripper_closed"]}
    return targets


def scripted_waiter(leader, follower, targets: dict, tolerance_deg: float = 0.5,
                    limit_s: float = DRY_RUN_LIMIT_S):
    """A stand-in for the person (dry runs and tests). At each step it moves the simulated
    leader (anything with a settable .pos) to targets[step] and presses Enter once the
    follower is within tolerance_deg of it on every joint."""
    asked = set()

    def wait_for(step: str, prompt: str):
        if step in asked:
            raise TeachError(f"dry run: the target for {step} wasn't accepted; check the targets")
        asked.add(step)
        target = {j: float(targets[step][j]) for j in JOINTS}
        leader.pos = dict(target)
        checks = 0

        class Arrived:
            def is_set(self) -> bool:
                nonlocal checks
                checks += 1
                if checks > limit_s * RATE_HZ:
                    raise TeachError(f"dry run: the follower never reached {step}")
                pos = follower.read()
                return all(abs(pos[j] - target[j]) < tolerance_deg for j in JOINTS)

        return Arrived()

    return wait_for


# --- teaching ----------------------------------------------------------------

def _same_pose(a: dict, b: dict) -> bool:
    return all(abs(a[j] - b[j]) < SAME_POSE_DEG for j in ARM_JOINTS)


def teach(leader, follower, speed_deg_s: float, wait_for=enter_waiter, log=print,
          with_tray: bool = False) -> dict:
    """Mirror leader -> follower at speed_deg_s; record the follower at each step.

    wait_for(step, prompt) returns an object whose is_set() is polled every tick and turns
    true when the person pressed Enter. with_tray also teaches above_tray. Returns the
    validated poses (raises PosesError)."""
    if not (_is_number(speed_deg_s) and 0 < speed_deg_s <= MAX_SPEED_DEG_S):
        raise ValueError(f"speed_deg_s must be in (0, {MAX_SPEED_DEG_S:g}] deg/s (plan section 17), "
                         f"got {speed_deg_s!r}")
    clock = getattr(follower, "clock", None) or RealClock()
    dt = 1.0 / RATE_HZ
    max_step = speed_deg_s * dt
    cmd = follower.read()
    recorded = {}
    last_name = None
    try:
        for name, hint in teach_steps(with_tray):
            prompt = f"[{name}] {hint}. Press Enter here when the follower is there: "
            while True:
                pressed = wait_for(name, prompt)
                while not pressed.is_set():
                    target = leader.read()
                    for j in ARM_JOINTS:
                        cmd[j] += max(-max_step, min(max_step, target[j] - cmd[j]))
                    cmd["gripper"] = target["gripper"]
                    follower.command(cmd)
                    clock.sleep(dt)
                pose = follower.read()
                if name in POSE_NAMES and last_name is not None and _same_pose(pose, recorded[last_name]):
                    log(f"  {name} is within {SAME_POSE_DEG:g} deg of {last_name} on every joint "
                        f"(an extra Enter?). Not recorded: move the arm there and press Enter again.")
                    continue
                break
            recorded[name] = pose
            if name in POSE_NAMES:
                last_name = name
            log(f"  {name}: " + ", ".join(f"{j} {pose[j]:.1f}" for j in JOINTS))
    finally:
        follower.command(follower.read())       # hold where it is
    poses = {name: recorded[name] for name in POSE_NAMES if name in recorded}
    poses["gripper_open"] = recorded["gripper_open"]["gripper"]
    poses["gripper_closed"] = recorded["gripper_closed"]["gripper"]
    return poses_mod.validate(poses, "tray" if with_tray else "hand")    # present is always taught


def save_with_backup(poses: dict, out: Path) -> Path | None:
    backup = None
    if out.exists():
        backup = REPO_ROOT / "data" / "poses_backup" / f"poses-{time.strftime('%Y%m%d-%H%M%S')}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, backup)
    poses_mod.save(poses, out)
    poses_mod.load(out)                         # reload check
    return backup


# --- command line ------------------------------------------------------------

def _error(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _close_all(*arms) -> None:
    """Close each arm that was opened; one failing close doesn't skip the others."""
    for arm in arms:
        if arm is None:
            continue
        try:
            arm.close()
        except Exception as e:  # noqa: BLE001
            _error(f"warning: closing {type(arm).__name__} failed: {type(e).__name__}: {e}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arm.teach",
                                     description="Teach the pick poses with the leader arm.")
    parser.add_argument("--out", type=Path,
                        help="where to save (default: config/poses.json; dry run: a new temporary "
                             "folder, and never config/poses.json)")
    parser.add_argument("--speed", type=float,
                        help="follower speed cap, deg/s: at most robot.yaml arm.move_speed_deg_s "
                             "(the default); first real runs: 25")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--with-tray", action="store_true",
                        help="also teach above_tray, for the stretch tray mode (python -m arm.pick run --to tray)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no hardware: a scripted person moves a simulated leader through the "
                             "sim's default poses")
    args = parser.parse_args(argv)

    try:
        cfg = load_arm_config(args.config, require=CONFIG_KEYS)
        speed = resolve_speed(args.speed, cfg)
        out = args.out
        if args.dry_run and out is not None and out.resolve() == poses_mod.DEFAULT_POSES_PATH.resolve():
            raise ValueError(f"a dry run never writes {poses_mod.DEFAULT_POSES_PATH}; "
                             f"pass another --out or leave it out")
    except ValueError as e:                     # includes pick.ConfigError
        _error(f"error: {e}")
        return 1

    leader = follower = None
    try:
        if args.dry_run:
            from .sim import DEFAULT_POSES, SimArm, SimLeader
            follower, leader = SimArm(), SimLeader()
            wait_for = scripted_waiter(leader, follower, dry_run_targets(DEFAULT_POSES, args.with_tray))
        else:
            from .real import RealArm, RealLeader
            # The leader is read-only, so open it first: if it fails, the follower is never touched.
            leader = RealLeader(port=cfg["leader_port"], robot_id=cfg["leader_id"])
            follower = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                               max_relative_target=cfg["max_relative_target_deg"])
            wait_for = enter_waiter
        print(f"teaching at {speed:g} deg/s" + (" (dry run, no hardware)" if args.dry_run else ""),
              flush=True)
        poses = teach(leader, follower, speed, wait_for=wait_for, with_tray=args.with_tray)
    except KeyboardInterrupt:
        _error("\nstopped; nothing saved")
        return 1
    except (poses_mod.PosesError, TeachError) as e:
        _error(f"not saved: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - a clear message, and the arms still get closed
        _error(f"not saved: {type(e).__name__}: {e}")
        return 1
    finally:
        _close_all(follower, leader)

    if out is None:
        out = (Path(tempfile.mkdtemp(prefix="arm-teach-dry-run-")) / "poses.json" if args.dry_run
               else poses_mod.DEFAULT_POSES_PATH)
    try:
        backup = save_with_backup(poses, out)
    except (OSError, poses_mod.PosesError) as e:
        _error(f"not saved: {type(e).__name__}: {e}")
        return 1
    print(f"saved and reloaded {out}" + (f" (old file backed up to {backup})" if backup else ""))
    if args.dry_run:
        print(f"dry run: {poses_mod.DEFAULT_POSES_PATH} was not touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
