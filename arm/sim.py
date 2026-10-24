"""Simulated SO-101 for --dry-run and tests.

Joints slide toward their targets. A closing claw stops on the "object" when the
arm is at the at_object pose. Options make the first N grasps miss, make the
object slip out while lifting, or freeze one joint (to test "stuck").

The sim runs on a virtual clock: pick.py sleeps through arm.clock, and each sleep
advances the simulation instead of waiting, so a whole pick takes milliseconds.
"""

from __future__ import annotations

import time

from . import poses as poses_mod
from .poses import ARM_JOINTS, JOINTS

# Plausible poses for running the sim before anything has been taught (task A3).
DEFAULT_POSES = {
    "ready":        {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0, "wrist_flex": 60.0, "wrist_roll": 0.0, "gripper": 30.0},
    "above_object": {"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 70.0, "wrist_roll": 0.0, "gripper": 60.0},
    "at_object":    {"shoulder_pan": 0.0, "shoulder_lift": 5.0, "elbow_flex": 25.0, "wrist_flex": 75.0, "wrist_roll": 0.0, "gripper": 60.0},
    "lift":         {"shoulder_pan": 0.0, "shoulder_lift": -60.0, "elbow_flex": 70.0, "wrist_flex": 40.0, "wrist_roll": 0.0, "gripper": 2.0},
    "above_tray":   {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 40.0, "wrist_flex": 95.0, "wrist_roll": 0.0, "gripper": 2.0},
    "gripper_open": 60.0,
    "gripper_closed": 2.0,
}

# How close (deg, on shoulder_lift / elbow_flex / wrist_flex) counts as "at" a pose.
# Pan and roll are ignored so --pan / --roll offsets still reach the object.
NEAR_DEG = 5.0


class SimClock:
    def __init__(self, arm: "SimArm", realtime: bool = False):
        self._arm = arm
        self.t = 0.0
        self.realtime = realtime    # also wait for real (interactive dry runs)

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self._arm.step(dt)
        self.t += dt
        if self.realtime:
            time.sleep(dt)


class SimArm:
    """Same interface as arm.real.RealArm: read(), command(targets), close(), clock."""

    def __init__(self, poses: dict | None = None, miss_first: int = 0, slip_on_lift: bool = False,
                 stall_joint: str | None = None, joint_speed_deg_s: float = 180.0,
                 grip_speed_s: float = 150.0, object_width: float = 15.0):
        if stall_joint is not None and stall_joint not in ARM_JOINTS:
            raise ValueError(f"stall_joint must be one of {ARM_JOINTS}, got {stall_joint!r}")
        self.poses = poses_mod.validate(poses or DEFAULT_POSES)
        self.pos = {j: float(self.poses["ready"][j]) for j in JOINTS}
        self.goal = dict(self.pos)
        self.clock = SimClock(self)
        self.misses_left = miss_first
        self.slip_on_lift = slip_on_lift
        self.stall_joint = stall_joint
        self.joint_speed = joint_speed_deg_s
        self.grip_speed = grip_speed_s
        self.object = "floor"           # floor | held | tray
        self.grasps = 0                 # closes that reached the object
        self._missing = False           # this close is a miss: the claw passes the object
        g_open, g_closed = self.poses["gripper_open"], self.poses["gripper_closed"]
        self._dir = 1.0 if g_open > g_closed else -1.0   # +1: opening increases the value
        self._object_stop = g_closed + self._dir * object_width

    # --- arm interface -------------------------------------------------------

    def read(self) -> dict:
        return dict(self.pos)

    def command(self, targets: dict) -> None:
        for j, v in targets.items():
            if j in self.goal:
                self.goal[j] = float(v)

    def close(self) -> None:
        pass

    # --- physics -------------------------------------------------------------

    def _near(self, pose_name: str) -> bool:
        pose = self.poses[pose_name]
        return all(abs(self.pos[j] - pose[j]) <= NEAR_DEG for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))

    def _opening_amount(self, value: float) -> float:
        """How far value is on the open side of the object stop."""
        return (value - self._object_stop) * self._dir

    def step(self, dt: float) -> None:
        for j in ARM_JOINTS:
            if j == self.stall_joint:
                continue
            self.pos[j] = _approach(self.pos[j], self.goal[j], self.joint_speed * dt)

        if self.object == "held" and self.slip_on_lift and self._near("lift"):
            self.object = "floor"       # slipped out: the claw now closes on nothing
            self._missing = True

        g = _approach(self.pos["gripper"], self.goal["gripper"], self.grip_speed * dt)
        if self.object == "held":
            if self._opening_amount(g) > 1.0:
                self.object = "tray" if self._near("above_tray") else "floor"
            else:
                g = self._object_stop if self._opening_amount(g) < 0 else g
        elif self.object == "floor" and self._opening_amount(g) < 0 <= self._opening_amount(self.pos["gripper"]):
            # The claw is closing past the object's width this step.
            if not self._missing and self._near("at_object"):
                if self.misses_left > 0:
                    self.misses_left -= 1
                    self._missing = True
                else:
                    self.object = "held"
                    self.grasps += 1
                    g = self._object_stop
        if self._missing and self._opening_amount(g) > 1.0:
            self._missing = False       # opened again: the next close is a new try
        self.pos["gripper"] = g


class SimLeader:
    """A leader arm that stays where it's put: set .pos to "move" it."""

    def __init__(self, pos: dict | None = None):
        self.pos = dict(pos or DEFAULT_POSES["ready"])

    def read(self) -> dict:
        return dict(self.pos)

    def close(self) -> None:
        pass


def _approach(value: float, goal: float, max_step: float) -> float:
    if abs(goal - value) <= max_step:
        return goal
    return value + max_step if goal > value else value - max_step
