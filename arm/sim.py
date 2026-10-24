"""Simulated SO-101 for --dry-run and tests.

Joints slide toward their targets. A closing claw stops on the "object" when the
arm is at the at_object pose. Options make the first N grasps miss, make the
object slip out while lifting, or freeze one joint (to test "stuck").

Grip: while the object is in the claw and the claw target is past it, the claw
stops at the object's width (object_width above closed), whatever the target, so
the full squeeze and the light hold (pick.LIGHT_SQUEEZE past the object) read the
same. Once the object is gone, the claw moves on to its target: fully closed when
squeezing, a few units closed from where it held the object in a light hold.

Hand-off: once the arm has arrived at the present pose (every joint at its goal)
and held the object there for take_after_s, the person takes it. The object is
gone, so the claw closes onto its target. never_take=True: nobody takes it (to
test the timeout). slip_at_present_after_s: the object falls out that long after
arriving, and the claw closes just the same. pick.py reports that as "dropped"
within pick.TAKE_SETTLE_S of arriving; later it can't be told from a take (the
known limit in plan 10.7).

The sim runs on a virtual clock: pick.py sleeps through arm.clock, and each sleep
advances the simulation instead of waiting, so a whole pick takes milliseconds.
"""

from __future__ import annotations

from . import poses as poses_mod
from .poses import ARM_JOINTS, JOINTS

# Plausible poses for running the sim before anything has been taught (task A3).
DEFAULT_POSES = {
    "ready":        {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0, "wrist_flex": 60.0, "wrist_roll": 0.0, "gripper": 30.0},
    "above_object": {"shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 40.0, "wrist_flex": 70.0, "wrist_roll": 0.0, "gripper": 60.0},
    "at_object":    {"shoulder_pan": 0.0, "shoulder_lift": 5.0, "elbow_flex": 25.0, "wrist_flex": 75.0, "wrist_roll": 0.0, "gripper": 60.0},
    "lift":         {"shoulder_pan": 0.0, "shoulder_lift": -60.0, "elbow_flex": 70.0, "wrist_flex": 40.0, "wrist_roll": 0.0, "gripper": 2.0},
    "present":      {"shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": 0.0, "wrist_flex": 10.0, "wrist_roll": 0.0, "gripper": 2.0},
    "above_tray":   {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 40.0, "wrist_flex": 95.0, "wrist_roll": 0.0, "gripper": 2.0},
    "gripper_open": 60.0,
    "gripper_closed": 2.0,
}

# How close (deg, on shoulder_lift / elbow_flex / wrist_flex) counts as "at" a pose.
# Pan and roll are ignored so --pan / --roll offsets still reach the object.
NEAR_DEG = 5.0


class SimClock:
    def __init__(self, arm: "SimArm"):
        self._arm = arm
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self._arm.step(dt)
        self.t += dt


class SimArm:
    """Same interface as arm.real.RealArm: read(), command(targets), close(), clock."""

    def __init__(self, poses: dict | None = None, miss_first: int = 0, slip_on_lift: bool = False,
                 stall_joint: str | None = None, take_after_s: float = 2.0, never_take: bool = False,
                 slip_at_present_after_s: float | None = None, joint_speed_deg_s: float = 180.0,
                 grip_speed_s: float = 150.0, object_width: float = 15.0):
        if stall_joint is not None and stall_joint not in ARM_JOINTS:
            raise ValueError(f"stall_joint must be one of {ARM_JOINTS}, got {stall_joint!r}")
        seconds = {"take_after_s": take_after_s}
        if slip_at_present_after_s is not None:
            seconds["slip_at_present_after_s"] = slip_at_present_after_s
        for name, value in seconds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ValueError(f"{name} must be a number of seconds >= 0, got {value!r}")
        self.poses = poses_mod.validate(poses or DEFAULT_POSES)
        self.pos = {j: float(self.poses["ready"][j]) for j in JOINTS}
        self.goal = dict(self.pos)
        self.clock = SimClock(self)
        self.misses_left = miss_first
        self.slip_on_lift = slip_on_lift
        self.stall_joint = stall_joint
        self.take_after_s = None if never_take else float(take_after_s)
        self.slip_at_present_after_s = slip_at_present_after_s
        self.presented_s = 0.0          # how long the object has been held at the present pose
        self.joint_speed = joint_speed_deg_s
        self.grip_speed = grip_speed_s
        self.object = "floor"           # floor | held | tray | hand (the person took it)
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
        if pose_name not in self.poses:  # present / above_tray: only one mode needs each
            return False
        pose = self.poses[pose_name]
        return all(abs(self.pos[j] - pose[j]) <= NEAR_DEG for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))

    def _arrived(self, pose_name: str) -> bool:
        """Every arm joint has reached its goal, and the goal is this pose: the arm has stopped there."""
        if pose_name not in self.poses:
            return False
        pose = self.poses[pose_name]
        return all(self.pos[j] == self.goal[j] and abs(self.goal[j] - pose[j]) < 1e-6 for j in ARM_JOINTS)

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

        if self.object == "held" and self._arrived("present"):
            self.presented_s += dt
            if self.slip_at_present_after_s is not None and self.presented_s >= self.slip_at_present_after_s:
                self.object = "floor"   # slipped out: the claw now closes on nothing
                self._missing = True
            elif self.take_after_s is not None and self.presented_s >= self.take_after_s:
                self.object = "hand"    # pulled out: nothing stops the claw now
        else:
            self.presented_s = 0.0

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
