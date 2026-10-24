"""Taught arm poses: the format of config/poses.json.

Written by arm/teach.py (task A3), read by arm/pick.py. Angles are in degrees as
LeRobot reports them (0 deg is the middle of each joint's calibrated range); the
gripper is on LeRobot's 0-100 scale.

    {
      "units": "degrees; gripper 0-100",
      "ready":        {"shoulder_pan": 0.0, "shoulder_lift": -95.0, ..., "gripper": 30.0},
      "above_object": {...}, "at_object": {...}, "lift": {...}, "above_tray": {...},
      "gripper_open": 60.0,
      "gripper_closed": 2.0
    }

Each pose holds all six joints. pick.py uses the five arm joints of each pose and
sets the claw itself from gripper_open / gripper_closed.
"""

from __future__ import annotations

import json
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
POSE_NAMES = ["ready", "above_object", "at_object", "lift", "above_tray"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSES_PATH = REPO_ROOT / "config" / "poses.json"


class PosesError(ValueError):
    pass


def validate(poses: dict) -> dict:
    """Check that every pose and both claw values are present and numeric. Returns poses."""
    for name in POSE_NAMES:
        pose = poses.get(name)
        if not isinstance(pose, dict):
            raise PosesError(f"pose '{name}' is missing")
        for joint in JOINTS:
            if not isinstance(pose.get(joint), (int, float)):
                raise PosesError(f"pose '{name}' has no number for '{joint}'")
    for key in ("gripper_open", "gripper_closed"):
        if not isinstance(poses.get(key), (int, float)):
            raise PosesError(f"'{key}' is missing")
    if abs(poses["gripper_open"] - poses["gripper_closed"]) < 10:
        raise PosesError("gripper_open and gripper_closed are less than 10 apart; re-teach them")
    return poses


def load(path: str | Path = DEFAULT_POSES_PATH) -> dict:
    with open(path) as f:
        return validate(json.load(f))


def save(poses: dict, path: str | Path = DEFAULT_POSES_PATH) -> None:
    validate(poses)
    out = {"units": "degrees; gripper 0-100"}
    out.update({name: {j: float(poses[name][j]) for j in JOINTS} for name in POSE_NAMES})
    out["gripper_open"] = float(poses["gripper_open"])
    out["gripper_closed"] = float(poses["gripper_closed"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(path)
