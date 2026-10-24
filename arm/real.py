"""The real SO-101 follower through LeRobot, with the interface pick.py expects.

Never calibrates: it refuses to run if the saved calibration (task A1) doesn't
match the arm. Leaves the torque on when closed, so the arm holds its pose.

No startup jolt (the same fix as safe_teleop.py): LeRobot turns the torque back on
at connect without touching Goal_Position, so the servos snap to a stale goal. Here
each motor's goal is first set to where it is, with a low torque limit, and the
limit then ramps up over RAMP_S seconds while the arm holds still.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import time

from .poses import JOINTS


START_FRACTION = 0.2       # torque limit when the torque comes on, as a fraction of the cap
RAMP_S = 1.5               # then ramped up to the cap over this long
RAMP_STEPS = 10
CONNECT_TRIES = 5          # a missed handshake on the bus is retried
FROZEN_READING = 0.0       # what read() reports for a frozen joint: it never moves, so any constant


def _torque_cap(motor: str) -> int:
    """Torque_Limit the arm ends up at: the same caps safe_teleop.py runs teleop with."""
    return 500 if motor in ("gripper", "shoulder_pan") else 1000


def _import_follower():
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # newer layout
    return SO101Follower, SO101FollowerConfig


class RealArm:
    def __init__(self, port: str, robot_id: str, max_relative_target: float, *,
                 frozen=(), ramp_s: float | None = None, sleep=None):
        """max_relative_target: LeRobot's per-command step cap in degrees, from robot.yaml
        arm.max_relative_target_deg. Required: None would mean no cap at all.
        frozen: arm joints (robot.yaml arm.frozen_joints) whose servos are never called: the
        bus is built without them, read() reports them at FROZEN_READING and command() drops
        them. Used for the follower's faulty wrist-roll servo (id 5)."""
        self.frozen = tuple(frozen)
        bad = [j for j in self.frozen if j not in JOINTS or j == "gripper"]
        if bad:
            raise ValueError(f"frozen joints must be arm joints (not the gripper), got {bad}")
        if (not isinstance(max_relative_target, numbers.Real) or isinstance(max_relative_target, bool)
                or not math.isfinite(max_relative_target) or max_relative_target <= 0):
            raise ValueError(f"max_relative_target must be a positive number of degrees, "
                             f"got {max_relative_target!r}")
        SO101Follower, SO101FollowerConfig = _import_follower()
        # float, never int: LeRobot's ensure_safe_goal_position accepts only a float or a dict and
        # raises TypeError on robot.yaml's integer `10` at the first command.
        kwargs = {"port": port, "id": robot_id, "max_relative_target": float(max_relative_target)}
        fields = {f.name for f in dataclasses.fields(SO101FollowerConfig)}
        if "disable_torque_on_disconnect" in fields:
            kwargs["disable_torque_on_disconnect"] = False
        if "use_degrees" in fields:
            kwargs["use_degrees"] = True
        self._keep_torque_via_config = "disable_torque_on_disconnect" in fields
        ramp_s = RAMP_S if ramp_s is None else ramp_s
        sleep = sleep or time.sleep
        self.robot = SO101Follower(SO101FollowerConfig(**kwargs))
        if self.frozen:
            self._drop_frozen_motors()
        self._hold_on_enable()
        self._connect(sleep)
        if not self.robot.is_calibrated:
            self.robot.bus.disconnect(disable_torque=False)
            raise RuntimeError(f"no matching calibration for '{robot_id}'; run task A1 (lerobot-calibrate) first")
        self._ramp_torque(ramp_s, sleep)

    def _drop_frozen_motors(self) -> None:
        """Rebuild the bus without the frozen motors, before it ever connects: no packet goes to
        them, not the handshake, not a read, not a write. They stay as they are, held by their
        gears (torque off after a power cycle)."""
        bus = self.robot.bus
        self.robot.bus = type(bus)(
            bus.port, {m: v for m, v in bus.motors.items() if m not in self.frozen},
            calibration={m: c for m, c in bus.calibration.items() if m not in self.frozen},
            protocol_version=bus.protocol_version)

    def _hold_on_enable(self) -> None:
        """Make every torque-enable hold the motors where they are, at a low torque limit."""
        bus = self.robot.bus
        enable = bus.enable_torque

        def enable_holding(motors=None, num_retry=0):
            if motors is None:
                names = list(bus.motors)
            elif isinstance(motors, (str, int)):
                names = [motors]
            else:
                names = list(motors)
            present = bus.sync_read("Present_Position", names, normalize=False, num_retry=10)
            for m in names:
                bus.write("Goal_Position", m, present[m], normalize=False, num_retry=10)
                bus.write("Torque_Limit", m, int(_torque_cap(m) * START_FRACTION), normalize=False, num_retry=10)
            enable(motors, num_retry)

        bus.enable_torque = enable_holding

    def _connect(self, sleep) -> None:
        for attempt in range(1, CONNECT_TRIES + 1):
            try:
                self.robot.connect(calibrate=False)
                return
            except (RuntimeError, ConnectionError) as error:
                if getattr(self.robot.bus, "is_connected", False):
                    self.robot.bus.disconnect(disable_torque=False)
                if attempt == CONNECT_TRIES:
                    raise RuntimeError(f"follower not answering after {CONNECT_TRIES} tries "
                                       f"(nothing moved): {str(error).splitlines()[0]}") from error
                sleep(0.5)

    def _ramp_torque(self, ramp_s: float, sleep) -> None:
        """Raise the torque limit from START_FRACTION to the cap; the goals equal the present
        positions, so nothing moves."""
        bus = self.robot.bus
        for step in range(1, RAMP_STEPS + 1):
            fraction = START_FRACTION + (1.0 - START_FRACTION) * step / RAMP_STEPS
            bus.sync_write("Torque_Limit", {m: int(_torque_cap(m) * fraction) for m in bus.motors},
                           normalize=False)
            sleep(ramp_s / RAMP_STEPS)

    def read(self) -> dict:
        obs = self.robot.get_observation()
        return {j: FROZEN_READING if j in self.frozen else float(obs[f"{j}.pos"]) for j in JOINTS}

    def command(self, targets: dict) -> None:
        action = {f"{j}.pos": float(v) for j, v in targets.items() if j in JOINTS and j not in self.frozen}
        if "gripper.pos" in action:
            action["gripper.pos"] = min(100.0, max(0.0, action["gripper.pos"]))
        self.robot.send_action(action)

    def close(self) -> None:
        if self._keep_torque_via_config:
            self.robot.disconnect()
        else:
            self.robot.bus.disconnect(disable_torque=False)


class RealLeader:
    """The SO-101 leader arm (read only; it's moved by hand)."""

    def __init__(self, port: str, robot_id: str):
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        fields = {f.name for f in dataclasses.fields(SO101LeaderConfig)}
        kwargs = {"port": port, "id": robot_id}
        if "use_degrees" in fields:
            kwargs["use_degrees"] = True
        self.leader = SO101Leader(SO101LeaderConfig(**kwargs))
        self.leader.connect(calibrate=False)
        if not self.leader.is_calibrated:
            self.leader.disconnect()
            raise RuntimeError(f"no matching calibration for '{robot_id}'; run task A1 (lerobot-calibrate) first")

    def read(self) -> dict:
        action = self.leader.get_action()
        return {j: float(action[f"{j}.pos"]) for j in JOINTS}

    def close(self) -> None:
        self.leader.disconnect()
