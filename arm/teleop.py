"""Teleop the follower with the leader, safely (SW1's rule: line-up table first).

    python -m arm.teleop [--speed DEG_S] [--config PATH] [--dry-run]

Use this instead of lerobot-teleoperate or the old safe_teleop.py: both build LeRobot's
SO101Follower with all six motors, so they call the faulty wrist-roll servo (id 5), which
browns out the whole bus.

1. The leader opens first (read-only): if it fails, the follower is never touched.
2. The follower opens through RealArm: the joints in robot.yaml arm.frozen_joints are never
   called, and there is no startup jolt (goal = where it is, then a torque ramp).
3. The live table (arm/align.py) shows both arms; Enter starts only once every joint matches.
4. The follower copies the leader. Each arm joint moves at most --speed deg/s (default and
   ceiling: robot.yaml arm.move_speed_deg_s), and its target never leads where the joint really
   is by more than LEAD_DEG: a follower that is blocked (pressing on the floor, held by a hand)
   doesn't wind up a big error and then snap when it comes free.
5. Ctrl-C stops between two commands: the arm is told to hold where it is (the claw keeps its
   grip) and keeps its torque on. A second Ctrl-C quits at once.

LeRobot's warning that it eased the claw (its step cap) is muted while teleoperating, as in
teaching, so the screen stays readable. An arm joint being clamped still shows.

--dry-run uses the simulated arms for a few simulated seconds and touches no hardware.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from . import align as align_mod
from .pick import DEFAULT_CONFIG_PATH, MAX_SPEED_DEG_S, RATE_HZ, RealClock, _is_number, load_arm_config, resolve_speed
from .poses import ARM_JOINTS, JOINTS
from .teach import CLAMP_LOGGER, CONFIG_KEYS, _QuietClawClamp, _close_all, _error

LEAD_DEG = 10.0             # an arm joint's target stays within this of where the joint really is
DRY_RUN_S = 3.0


def follow(cmd: dict, target: dict, now: dict, max_step: float, frozen=()) -> dict:
    """One frame: each arm joint of cmd moves at most max_step towards target, and then stays
    within LEAD_DEG of `now` (the follower's reading); the claw goes to target's (LeRobot's
    max_relative_target eases it). Frozen joints aren't touched. Updates cmd and returns it."""
    for j in ARM_JOINTS:
        if j in frozen:
            continue
        stepped = cmd[j] + max(-max_step, min(max_step, target[j] - cmd[j]))
        cmd[j] = max(now[j] - LEAD_DEG, min(now[j] + LEAD_DEG, stepped))
    cmd["gripper"] = target["gripper"]
    return cmd


def teleop(leader, follower, speed_deg_s: float, *, stop=lambda: False, max_s: float | None = None) -> None:
    """Follower copies leader (module docstring) until stop() is true or max_s (simulated or
    wall) seconds pass. The caller must have lined the arms up first. Ends by telling the
    follower to hold where it is."""
    if not (_is_number(speed_deg_s) and 0 < speed_deg_s <= MAX_SPEED_DEG_S):
        raise ValueError(f"speed_deg_s must be in (0, {MAX_SPEED_DEG_S:g}] deg/s, got {speed_deg_s!r}")
    clock = getattr(follower, "clock", None) or RealClock()
    dt = 1.0 / RATE_HZ
    max_step = speed_deg_s * dt
    frozen = tuple(getattr(follower, "frozen", ()))
    cmd = follower.read()
    t0 = clock.now()
    quiet = _QuietClawClamp()
    clamp_log = logging.getLogger(CLAMP_LOGGER)
    clamp_log.addFilter(quiet)
    try:
        while not stop() and (max_s is None or clock.now() - t0 < max_s):
            target = leader.read()
            now = follower.read()
            follow(cmd, target, now, max_step, frozen)
            if stop():                                  # Ctrl-C landed during the reads: send nothing more
                break
            follower.command({j: v for j, v in cmd.items() if j not in frozen})
            clock.sleep(dt)
    finally:
        try:
            here = follower.read()              # hold the arm where it is; the claw keeps its grip
            follower.command({**{j: here[j] for j in ARM_JOINTS if j not in frozen}, "gripper": cmd["gripper"]})
        finally:
            clamp_log.removeFilter(quiet)


def _catch_ctrl_c(presses: list):
    """Ctrl-C only counts (checked between commands); a second press raises at once.
    Returns the function that puts the old handler back."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def press(*_):
        presses.append(1)
        if len(presses) > 1:
            raise KeyboardInterrupt

    old = signal.signal(signal.SIGINT, press)
    return lambda: signal.signal(signal.SIGINT, old)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arm.teleop",
                                     description="Teleop the follower with the leader (frozen wrist, no jolt, "
                                                 "line-up table first).")
    parser.add_argument("--speed", type=float,
                        help="follower speed cap, deg/s: at most robot.yaml arm.move_speed_deg_s (the default)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help=f"no hardware: simulated arms for {DRY_RUN_S:g} simulated seconds")
    args = parser.parse_args(argv)

    try:
        cfg = load_arm_config(args.config, require=CONFIG_KEYS)
        speed = resolve_speed(args.speed, cfg)
    except ValueError as e:                     # includes pick.ConfigError
        _error(f"error: {e}")
        return 1

    leader = follower = None
    presses = []
    restore = None
    try:
        if args.dry_run:
            from .sim import SimArm, SimLeader
            follower, leader = SimArm(), SimLeader()
            align_mod.scripted_align(leader, follower)
            teleop(leader, follower, speed, max_s=DRY_RUN_S)
            print(f"dry run: teleop ran {DRY_RUN_S:g} simulated seconds at {speed:g} deg/s; no hardware touched")
            return 0
        from .real import RealArm, RealLeader
        leader = RealLeader(port=cfg["leader_port"], robot_id=cfg["leader_id"])
        follower = RealArm(port=cfg["follower_port"], robot_id=cfg["follower_id"],
                           max_relative_target=cfg["max_relative_target_deg"], frozen=cfg["frozen_joints"])
        align_mod.live_align(leader, follower)  # the follower holds still until the arms match
        restore = _catch_ctrl_c(presses)
        frozen = ", ".join(cfg["frozen_joints"]) or "none"
        print(f"Teleop running at {speed:g} deg/s (frozen, never moved: {frozen}). "
              f"Ctrl-C stops; the follower holds where it is.", flush=True)
        teleop(leader, follower, speed, stop=lambda: bool(presses))
        print("\nstopped: the follower holds where it is, torque on. Rest it or hold it before you "
              "switch it off.", flush=True)
        return 0
    except KeyboardInterrupt:
        _error("\nstopped" + ("; the follower holds where it is" if follower is not None else "; nothing moved"))
        return 0
    except align_mod.AlignError as e:
        _error(f"not started: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - a clear message, and the arms still get closed
        _error(f"stopped: {type(e).__name__}: {e}")
        return 1
    finally:
        if restore is not None:
            restore()
        _close_all(follower, leader)


if __name__ == "__main__":
    sys.exit(main())
