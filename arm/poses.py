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

Suction mode (pick_to_hand(..., grasp="suction"), the flat-phone path) uses its own four
poses, suction_above, suction_press, suction_lift and suction_present, because the cup
reaches the phone at a different arm pose than the claw. They sit beside the claw poses in
the same file. The mode needs only `ready` and those four: the claw poses and the
gripper_open / gripper_closed values may be missing (they are checked if they are there).
The claw is never moved by a suction pick: it stays at the gripper value taught in
suction_press.
"""

from __future__ import annotations

import json
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
BASE_POSES = ["ready", "above_object", "at_object", "lift"]            # every claw mode
SUCTION_POSES = ["suction_above", "suction_press", "suction_lift", "suction_present"]
MODE_POSES = {"hand": ["present"], "tray": ["above_tray"], "suction": SUCTION_POSES}
POSE_NAMES = [*BASE_POSES, "present", "above_tray", *SUCTION_POSES]    # all of them, in teaching order
TEACH_HINT = {"hand": "python -m arm.teach", "tray": "python -m arm.teach --with-tray",
              "suction": "python -m arm.teach --grasp suction"}

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
    """Check the poses every claw mode needs, the ones `mode` ("hand", "tray" or "suction")
    needs, any other pose that is there, and both claw values. Returns poses.

    Suction mode needs only `ready` and the suction_* poses; the claw poses and claw values
    are then checked only if they are there (a half-present pair is an error)."""
    if mode is not None and mode not in MODE_POSES:
        raise ValueError(f"mode must be one of {sorted(MODE_POSES)}, got {mode!r}")
    for name in MODE_POSES.get(mode, []):
        if name not in poses:
            raise PosesError(f"pose '{name}' is missing; {mode} mode needs it "
                             f"(teach it with {TEACH_HINT[mode]})")
    required = ["ready"] if mode == "suction" else BASE_POSES
    for name in POSE_NAMES:
        if name in required or name in poses:
            _check_pose(poses, name)
    claw_keys = ("gripper_open", "gripper_closed")
    if mode != "suction" or any(key in poses for key in claw_keys):
        for key in claw_keys:
            if not isinstance(poses.get(key), (int, float)):
                raise PosesError(f"'{key}' is missing")
        if abs(poses["gripper_open"] - poses["gripper_closed"]) < MIN_CLAW_GAP:
            raise PosesError(f"gripper_open and gripper_closed are less than {MIN_CLAW_GAP:g} apart; "
                             f"re-teach them")
    return poses


def load(path: str | Path = DEFAULT_POSES_PATH, mode: str | None = None) -> dict:
    with open(path) as f:
        return validate(json.load(f), mode)


def load_partial(path: str | Path) -> dict:
    """What is already in a poses file, unchecked: {} if there is no file. For teaching one
    grasp type into a file that holds another's poses (the result is validated on save).
    Raises PosesError if the file is not a JSON object."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as e:
        raise PosesError(f"{path} is not valid JSON ({e}); fix or move it before teaching into it") from e
    if not isinstance(data, dict):
        raise PosesError(f"{path} does not hold a JSON object; fix or move it before teaching into it")
    data.pop("units", None)
    return data


def save(poses: dict, path: str | Path = DEFAULT_POSES_PATH, mode: str | None = None) -> None:
    validate(poses, mode)
    out = {"units": "degrees; gripper 0-100"}
    out.update({name: {j: float(poses[name][j]) for j in JOINTS} for name in POSE_NAMES if name in poses})
    for key in ("gripper_open", "gripper_closed"):
        if key in poses:
            out[key] = float(poses[key])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    tmp.replace(path)
