"""Pre-flight check to run before teaching or running the SO-101 arms (task A8).

    python -m arm.check [--config PATH] [--poses PATH] [--to hand|tray]

Prints a checklist and exits 0 if nothing failed, 1 otherwise:

    ✓ passed     ✗ failed (exit 1)     ! warning (exit 0)     - information only

It only reads files and checks that paths exist. It never moves an arm and never opens
a serial port or a camera, so it is safe with the arms unpowered or already in use.

Checks: robot.yaml (the arm settings and, for hand mode, the handoff settings, through
arm/pick.py's loaders); both arm ports exist and differ; LeRobot's calibration file for
each arm exists; the follower's and the leader's calibrated spans agree (plan section 20:
a short sweep makes the same angle a different pose on the two arms); poses.json is valid
for the mode; the cameras robot.yaml lists (information only).

LeRobot's calibration files (0.6.2, robots/robot.py, teleoperators/teleoperator.py and
utils/constants.py): <calibration root>/robots/so_follower/<follower_id>.json for the
follower and <calibration root>/teleoperators/so_leader/<leader_id>.json for the leader.
The root is $HF_LEROBOT_CALIBRATION, else $HF_LEROBOT_HOME/calibration, else
<HF_HOME>/lerobot/calibration (HF_HOME defaults to ~/.cache/huggingface).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import poses as poses_mod
from .pick import (DEFAULT_CONFIG_PATH, ConfigError, _is_number, _load_config, load_arm_config,
                   load_camera_indices, load_handoff_grip, load_handoff_timeout)
from .poses import JOINTS
from .teach import CONFIG_KEYS

OK, FAIL, WARN, INFO = "ok", "fail", "warn", "info"
SYMBOL = {OK: "✓", FAIL: "✗", WARN: "!", INFO: "-"}

TICKS_PER_TURN = 4096           # one turn of an STS3215 servo
SPAN_TOLERANCE = 0.15           # spans further apart than this, as a share of the larger span, warn
RANGE_JOINTS = [j for j in JOINTS if j != "gripper"]    # LeRobot maps the gripper 0-100 on both arms
BACKUP_DIR = "data/calibration"     # plan section 16 rule 7: back calibration files up here first


@dataclass
class Check:
    status: str
    text: str
    hints: list[str] = field(default_factory=list)


class CalibrationError(ValueError):
    pass


def _problem(e: Exception) -> str:
    return str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"


# --- robot.yaml --------------------------------------------------------------

def check_config(config: Path, mode: str) -> tuple[list[Check], dict | None]:
    """The robot.yaml checks, and the validated `arm:` section (None if it isn't valid)."""
    if not config.exists():
        return [Check(FAIL, f"robot.yaml not found: {config}",
                      ["it comes from task S6 (plan section 9.1); pass --config to use another file"])], None
    checks = []
    try:
        arm = load_arm_config(config, require=CONFIG_KEYS)
        checks.append(Check(OK, f"robot.yaml arm settings valid (max step {arm['max_relative_target_deg']:g} deg, "
                                f"speed {arm['move_speed_deg_s']:g} deg/s)"))
    except Exception as e:  # noqa: BLE001 - ConfigError, or YAML that doesn't parse
        arm = None
        checks.append(Check(FAIL, f"robot.yaml arm settings: {_problem(e)}"))
    try:
        timeout = load_handoff_timeout(config)
        grip = load_handoff_grip(config)
        checks.append(Check(OK, f"robot.yaml handoff settings valid (timeout {timeout:g} s, grip {grip})"))
    except Exception as e:  # noqa: BLE001
        # Only hand mode reads them, so in tray mode a bad value is a warning.
        checks.append(Check(FAIL if mode == "hand" else WARN, f"robot.yaml handoff settings: {_problem(e)}"
                            + ("" if mode == "hand" else " (only --to hand uses them)")))
    return checks, arm


# --- ports -------------------------------------------------------------------

def check_ports(arm: dict) -> list[Check]:
    """Both arm ports exist as paths (never opened) and are different."""
    checks = []
    for role in ("follower", "leader"):
        port = arm[f"{role}_port"]
        if os.path.exists(port):
            checks.append(Check(OK, f"{role} port {port} exists"))
        else:
            checks.append(Check(FAIL, f"{role} port {port} does not exist",
                                ["plug it in or rerun lerobot-find-port; macOS can renumber ports after "
                                 f"replugging. Then update arm.{role}_port in robot.yaml"]))
    if arm["follower_port"] == arm["leader_port"]:
        checks.append(Check(FAIL, f"arm.follower_port and arm.leader_port are both {arm['follower_port']}",
                            ["the 12 V follower and the 5 V leader are different devices; "
                             "rerun lerobot-find-port with one arm unplugged at a time"]))
    return checks


# --- calibration -------------------------------------------------------------

def calibration_root() -> Path:
    """The folder LeRobot reads calibration files from (lerobot/utils/constants.py)."""
    explicit = os.environ.get("HF_LEROBOT_CALIBRATION")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_LEROBOT_HOME")
    if home:
        return Path(home).expanduser() / "calibration"
    hf_home = os.path.expandvars(os.path.expanduser(os.environ.get(
        "HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
                                "huggingface"))))
    return Path(hf_home) / "lerobot" / "calibration"


def calibration_path(root: Path, role: str, device_id: str) -> Path:
    """Where LeRobot keeps the calibration of the "follower" (a so_follower robot) or the
    "leader" (a so_leader teleoperator) with this id."""
    kind, name = ("robots", "so_follower") if role == "follower" else ("teleoperators", "so_leader")
    return root / kind / name / f"{device_id}.json"


def read_calibration(path: Path) -> dict[str, tuple[float, float]]:
    """{joint: (range_min, range_max)}, in servo ticks, from a LeRobot calibration file.
    Raises CalibrationError if it can't be read or a joint's range is missing or empty."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise CalibrationError(f"{path} can't be read: {_problem(e)}") from e
    if not isinstance(data, dict):
        raise CalibrationError(f"{path} is not a calibration file (expected an object of joints)")
    ranges = {}
    for joint in JOINTS:
        entry = data.get(joint)
        low = entry.get("range_min") if isinstance(entry, dict) else None
        high = entry.get("range_max") if isinstance(entry, dict) else None
        if not (_is_number(low) and _is_number(high)):
            raise CalibrationError(f"{path} has no range_min and range_max for '{joint}'")
        if high <= low:
            raise CalibrationError(f"{path}: '{joint}' has range_max {high:g} not above range_min {low:g}")
        ranges[joint] = (low, high)
    return ranges


def _calibrate_command(role: str, arm: dict) -> str:
    flag = "robot" if role == "follower" else "teleop"
    return (f"lerobot-calibrate --{flag}.type=so101_{role} --{flag}.port={shlex.quote(arm[f'{role}_port'])} "
            f"--{flag}.id={shlex.quote(arm[f'{role}_id'])}")


def check_calibration(arm: dict, root: Path) -> tuple[list[Check], dict[str, dict]]:
    """Each arm's calibration file exists and reads. Also returns the ranges that read."""
    checks, loaded = [], {}
    for role, label in (("follower", "follower (so_follower robot)"), ("leader", "leader (so_leader teleoperator)")):
        path = calibration_path(root, role, arm[f"{role}_id"])
        if not path.is_file():
            checks.append(Check(FAIL, f"{label} calibration not found: {path}",
                                [f"calibrate it with: {_calibrate_command(role, arm)}",
                                 "(sweep every joint fully; plan section 20)"]))
            continue
        try:
            loaded[role] = read_calibration(path)
        except CalibrationError as e:
            checks.append(Check(FAIL, f"{label} calibration: {e}"))
            continue
        checks.append(Check(OK, f"{label} calibration found: {path}"))
    return checks, loaded


def _degrees(ticks: float) -> str:
    return f"{ticks * 360 / TICKS_PER_TURN:.0f}°"


def check_ranges(arm: dict, ranges: dict[str, dict], root: Path) -> list[Check]:
    """Warn for each joint (not the gripper) whose follower and leader spans differ by more
    than SPAN_TOLERANCE, and say how to fix the arm with the shorter span."""
    spans = {role: {j: high - low for j, (low, high) in ranges[role].items()} for role in ("follower", "leader")}
    checks, short_on = [], {"follower": [], "leader": []}
    for joint in RANGE_JOINTS:
        follower, leader = spans["follower"][joint], spans["leader"][joint]
        gap = abs(follower - leader) / max(follower, leader)
        if gap > SPAN_TOLERANCE:
            short_on["follower" if follower < leader else "leader"].append(joint)
            checks.append(Check(WARN, f"{joint} calibrated span differs: follower {_degrees(follower)}, "
                                      f"leader {_degrees(leader)} ({gap:.0%} apart, over {SPAN_TOLERANCE:.0%})"))
    if not checks:
        return [Check(OK, f"calibrated spans agree within {SPAN_TOLERANCE:.0%} on every joint "
                          f"(gripper skipped: LeRobot maps it 0-100 on both arms)")]
    for role in ("leader", "follower"):
        joints = short_on[role]
        if not joints:
            continue
        device_id = arm[f"{role}_id"]
        checks.append(Check(WARN, f"recalibrate the {role}: its sweep of {', '.join(joints)} was probably short",
                            ["the same angle means a different pose on the two arms, so teleop and teaching are "
                             "offset and the other arm may not reach its full range",
                             f"1. back up the file: mkdir -p {BACKUP_DIR} && cp "
                             f"{shlex.quote(str(calibration_path(root, role, device_id)))} "
                             f"{BACKUP_DIR}/{device_id}-$(date +%Y%m%d-%H%M%S).json",
                             f"2. recalibrate: {_calibrate_command(role, arm)}",
                             "3. at the prompts, sweep every joint fully, end to end, including the folded rest pose"]))
    return checks


# --- poses -------------------------------------------------------------------

def check_poses(path: Path, mode: str) -> Check:
    """poses.json exists and validates for the mode."""
    if not path.exists():
        return Check(FAIL, f"poses file not found: {path}", [f"run {poses_mod.TEACH_HINT[mode]}"])
    try:
        with open(path) as f:
            poses = json.load(f)
        if not isinstance(poses, dict):
            raise poses_mod.PosesError("the top level must be an object of poses")
        poses_mod.validate(poses, mode)
    except (OSError, ValueError) as e:      # includes JSON errors and PosesError
        return Check(FAIL, f"poses file {path}: {_problem(e)}", [f"re-teach with {poses_mod.TEACH_HINT[mode]}"])
    names = [n for n in poses_mod.POSE_NAMES if n in poses]
    return Check(OK, f"poses file valid for {mode} mode ({path}: {', '.join(names)})")


# --- cameras -----------------------------------------------------------------

def check_cameras(config: Path) -> list[Check]:
    """The cameras robot.yaml lists, for information only: they are not opened."""
    try:
        cams = _load_config(config).get("cameras")
    except Exception:  # noqa: BLE001 - the robot.yaml checks already say why it doesn't load
        return []
    names = list(cams) if isinstance(cams, dict) else []
    try:
        if not names:
            raise ConfigError("robot.yaml has no cameras listed")
        indices = load_camera_indices(config, names)
    except ConfigError as e:
        return [Check(WARN, f"cameras: {e}")]
    listing = ", ".join(f"{name}={index}" for name, index in indices.items())
    return [Check(INFO, f"cameras in robot.yaml (not opened): {listing}",
                  ["macOS can renumber cameras after replugging: check the indices at the start of each session "
                   "(plan section 20)"])]


# --- run and print -----------------------------------------------------------

def run_checks(config: Path, poses: Path, mode: str, root: Path | None = None) -> list[Check]:
    """Every check, in order. root is LeRobot's calibration folder (default: where LeRobot looks)."""
    root = calibration_root() if root is None else root
    checks, arm = check_config(config, mode)
    if arm is None:
        checks.append(Check(INFO, "ports and calibration not checked: fix the robot.yaml arm settings first"))
    else:
        checks += check_ports(arm)
        found, ranges = check_calibration(arm, root)
        checks += found
        if len(ranges) == 2:
            checks += check_ranges(arm, ranges, root)
    checks.append(check_poses(poses, mode))
    if config.exists():
        checks += check_cameras(config)
    return checks


def format_report(checks: list[Check], mode: str) -> str:
    lines = [f"Arm pre-flight check, {mode} mode (reads files only; nothing moves, no port or camera is opened)", ""]
    for check in checks:
        lines.append(f"{SYMBOL[check.status]} {check.text}")
        lines += [f"    {hint}" for hint in check.hints]
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append("")
    if failed:
        lines.append(f"Not ready: {failed} failed, {warned} warning(s). Fix the {SYMBOL[FAIL]} items first.")
    elif warned:
        lines.append(f"Ready, with {warned} warning(s): read the {SYMBOL[WARN]} items above.")
    else:
        lines.append("Ready: every check passed.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arm.check",
                                     description="Pre-flight check before teaching or running the arms. "
                                                 "Reads files only: it never moves an arm or opens a port or camera.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--poses", type=Path, default=poses_mod.DEFAULT_POSES_PATH)
    parser.add_argument("--to", choices=sorted(poses_mod.MODE_POSES), default="hand",
                        help="the mode to check the poses (and handoff settings) for; default hand")
    args = parser.parse_args(argv)
    checks = run_checks(args.config, args.poses, args.to)
    print(format_report(checks, args.to), flush=True)
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
