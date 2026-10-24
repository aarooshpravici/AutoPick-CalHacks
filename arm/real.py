"""The real SO-101 follower through LeRobot, with the interface pick.py expects.

Never calibrates: it refuses to run if the saved calibration (task A1) doesn't
match the arm. Leaves the torque on when closed, so the arm holds its pose.
"""

from __future__ import annotations

import dataclasses

from .poses import JOINTS


def _import_follower():
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # newer layout
    return SO101Follower, SO101FollowerConfig


class RealArm:
    def __init__(self, port: str, robot_id: str, max_relative_target: float = 10):
        SO101Follower, SO101FollowerConfig = _import_follower()
        kwargs = {"port": port, "id": robot_id, "max_relative_target": max_relative_target}
        fields = {f.name for f in dataclasses.fields(SO101FollowerConfig)}
        if "disable_torque_on_disconnect" in fields:
            kwargs["disable_torque_on_disconnect"] = False
        if "use_degrees" in fields:
            kwargs["use_degrees"] = True
        self._keep_torque_via_config = "disable_torque_on_disconnect" in fields
        self.robot = SO101Follower(SO101FollowerConfig(**kwargs))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.bus.disconnect(disable_torque=False)
            raise RuntimeError(f"no matching calibration for '{robot_id}'; run task A1 (lerobot-calibrate) first")

    def read(self) -> dict:
        obs = self.robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS}

    def command(self, targets: dict) -> None:
        action = {f"{j}.pos": float(v) for j, v in targets.items() if j in JOINTS}
        if "gripper.pos" in action:
            action["gripper.pos"] = min(100.0, max(0.0, action["gripper.pos"]))
        self.robot.send_action(action)

    def close(self) -> None:
        if self._keep_torque_via_config:
            self.robot.disconnect()
        else:
            self.robot.bus.disconnect(disable_torque=False)
