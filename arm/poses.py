"""Taught arm poses: the format of config/poses.json.

Written by arm/teach.py (task A3), read by arm/pick.py. Angles are in degrees as
LeRobot reports them (0 deg is the middle of each joint's calibrated range); the
gripper is on LeRobot's 0-100 scale.

    {
      "units": "degrees; gripper 0-100",
      "ready":        {"shoulder_pan": 0.0, "shoulder_lift": -95.0, ..., "gripper": 30.0},
      "above_object": {...}, "at_object": {...}, "lift": {...}, "present": {...},
      "above_tray": {...},
      "gripper_open": 60.0,
      "gripper_closed": 2.0
    }

Each pose holds all six joints. pick.py uses the five arm joints of each pose and
sets the claw itself from gripper_open / gripper_closed.

Hand mode (pick_to_hand, the default) needs `present`; tray mode (pick_to_tray,
stretch) needs `above_tray`. Either may be missing from the file, but a mode whose
pose is missing is refused before the arm moves.
"""

from __future__ import annotations

import json
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
BASE_POSES = ["ready", "above_object", "at_object", "lift"]            # every mode
MODE_POSES = {"hand": ["present"], "tray": ["above_tray"]}
POSE_NAMES = [*BASE_POSES, "present", "above_tray"]                    # all of them, in teaching order
TEACH_HINT = {"hand": "python -m arm.teach", "tray": "python -m arm.teach --with-tray"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSES_PATH = REPO_ROOT / "config" / "poses.json"
MIN_CLAW_GAP = 10.0         # gripper_open and gripper_closed at least this far apart (0-100 scale)


class PosesError(ValueError):
    pass


def _check_pose(poses: dict, name: str) -> None:
    pose = poses.get(name)
    if not isinstance(pose, dict):
        raise PosesError(f"pose '{name}' is missing")
    for joint in JOINTS:
        if not isinstance(pose.get(joint), (int, float)):
            raise PosesError(f"pose '{name}' has no number for '{joint}'")


def validate(poses: dict, mode: str | None = None) -> dict:
    """Check the poses every mode needs, the ones `mode` ("hand" or "tray") needs, any
    other pose that is there, and both claw values. Returns poses."""
    if mode is not None and mode not in MODE_POSES:
        raise ValueError(f"mode must be one of {sorted(MODE_POSES)}, got {mode!r}")
    for name in MODE_POSES.get(mode, []):
        if name not in poses:
            raise PosesError(f"pose '{name}' is missing; {mode} mode needs it "
                             f"(teach it with {TEACH_HINT[mode]})")
    for name in POSE_NAMES:
        if name in BASE_POSES or name in poses:
            _check_pose(poses, name)
    for key in ("gripper_open", "gripper_closed"):
        if not isinstance(poses.get(key), (int, float)):
            raise PosesError(f"'{key}' is missing")
    if abs(poses["gripper_open"] - poses["gripper_closed"]) < MIN_CLAW_GAP:
        raise PosesError(f"gripper_open and gripper_closed are less than {MIN_CLAW_GAP:g} apart; re-teach them")
    return poses


def load(path: str | Path = DEFAULT_POSES_PATH, mode: str | None = None) -> dict:
    with open(path) as f:
        return validate(json.load(f), mode)


def save(poses: dict, path: str | Path = DEFAULT_POSES_PATH) -> None:
    validate(poses)
    out = {"units": "degrees; gripper 0-100"}
    out.update({name: {j: float(poses[name][j]) for j in JOINTS} for name in POSE_NAMES if name in poses})
    out["gripper_open"] = float(poses["gripper_open"])
    out["gripper_closed"] = float(poses["gripper_closed"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(path)
