"""Tests for the pre-flight check (task A8). Temporary files and fake device paths: no hardware."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import arm.check as check_mod
from arm import poses as poses_mod
from arm.check import FAIL, INFO, OK, WARN, calibration_path, calibration_root, main, run_checks
from arm.sim import DEFAULT_POSES

REPO_ROOT = Path(__file__).resolve().parent.parent

FOLLOWER_ID = "so101_follower_arm"
LEADER_ID = "so101_leader_arm"
TICKS = 4096 / 360          # servo ticks per degree
# Spans in ticks, as LeRobot saves them (range_max - range_min); wrist_roll is always a full turn.
SPANS = {"shoulder_pan": 2034, "shoulder_lift": 2384, "elbow_flex": 2217, "wrist_flex": 2010,
         "wrist_roll": 4095, "gripper": 1298}
WRIST_177 = round(177 * TICKS)      # 2014 ticks
WRIST_126 = round(126 * TICKS)      # 1433 ticks
DROP = object()                     # write_config(...=DROP) leaves the key out


def write_calibration(path, spans=SPANS, **joint_spans):
    """A LeRobot calibration file; each joint's range starts at 500 ticks."""
    spans = {**spans, **joint_spans}
    data = {joint: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 500,
                    "range_max": 500 + span}
            for i, (joint, span) in enumerate(spans.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))
    return path


class Setup:
    """A complete, passing setup in a temporary folder: ports, robot.yaml, both calibration
    files and poses.json. Tests break one part at a time."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.follower_port = tmp_path / "ttyFOLLOWER"
        self.leader_port = tmp_path / "ttyLEADER"
        self.follower_port.touch()
        self.leader_port.touch()
        self.root = tmp_path / "calibration"
        monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(self.root))       # where main() looks
        self.config = tmp_path / "robot.yaml"
        self.poses = tmp_path / "poses.json"
        self.write_config()
        write_calibration(self.cal("follower"))
        write_calibration(self.cal("leader"))
        poses_mod.save(DEFAULT_POSES, self.poses)

    def cal(self, role):
        return calibration_path(self.root, role, {"follower": FOLLOWER_ID, "leader": LEADER_ID}[role])

    def write_config(self, arm=None, handoff=None, cameras=None, **arm_overrides):
        data = {
            "arm": {"follower_port": str(self.follower_port), "leader_port": str(self.leader_port),
                    "follower_id": FOLLOWER_ID, "leader_id": LEADER_ID,
                    "max_relative_target_deg": 10, "move_speed_deg_s": 45},
            "cameras": {"wrist": 1, "scene": 2, "mast_top": 3} if cameras is None else cameras,
            "handoff": {"timeout_s": 60, "grip": "squeeze"} if handoff is None else handoff,
        }
        data["arm"].update(arm or {})
        for key, value in arm_overrides.items():
            if value is DROP:
                data["arm"].pop(key, None)
            else:
                data["arm"][key] = value
        self.config.write_text(yaml.safe_dump(data, sort_keys=False))

    def checks(self, mode="hand"):
        return run_checks(self.config, self.poses, mode, self.root)

    def run(self, capsys, *extra):
        """main() the way the person runs it; returns (exit code, printed text)."""
        code = main(["--config", str(self.config), "--poses", str(self.poses), *extra])
        return code, capsys.readouterr().out


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return Setup(tmp_path, monkeypatch)


def statuses(checks):
    return [c.status for c in checks]


def find(checks, text, status=None):
    """The checks whose text contains `text` (and have this status, if given)."""
    return [c for c in checks if text in c.text and (status is None or c.status == status)]


def one(checks, text, status=None):
    found = find(checks, text, status)
    assert len(found) == 1, [(c.status, c.text) for c in checks]
    return found[0]


def everything(check):
    return " ".join([check.text, *check.hints])


# --- everything passes -----------------------------------------------------------------

def test_a_complete_setup_passes_every_check_and_exits_0(setup, capsys):
    checks = setup.checks()
    assert set(statuses(checks)) == {OK, INFO}          # INFO: the camera list
    code, out = setup.run(capsys)
    assert code == 0
    assert "✗" not in out and "\n! " not in out
    assert out.count("✓") == len([c for c in checks if c.status == OK])
    assert "Ready: every check passed." in out


def test_the_report_marks_each_line_with_its_symbol(setup, capsys):
    setup.poses.unlink()
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126)
    code, out = setup.run(capsys)
    lines = out.splitlines()
    assert any(line.startswith("✓ robot.yaml arm settings valid") for line in lines)
    assert any(line.startswith("✗ poses file not found") for line in lines)
    assert any(line.startswith("! wrist_flex calibrated span differs") for line in lines)
    assert any(line.startswith("- cameras in robot.yaml") for line in lines)
    assert code == 1


# --- exit codes ------------------------------------------------------------------------

def test_exit_code_is_0_when_there_are_only_warnings(setup, capsys):
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126)
    code, out = setup.run(capsys)
    assert code == 0
    assert "\n! wrist_flex" in out
    assert "Ready, with 2 warning(s)" in out


def test_exit_code_is_1_when_anything_fails(setup, capsys):
    setup.follower_port.unlink()
    code, out = setup.run(capsys)
    assert code == 1
    assert "Not ready: 1 failed" in out


def test_a_failure_wins_over_warnings(setup, capsys):
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126)
    setup.poses.unlink()
    code, out = setup.run(capsys)
    assert code == 1
    assert "Not ready: 1 failed, 2 warning(s)" in out


def test_the_mode_flag_picks_the_poses_and_handoff_checks(setup, capsys):
    setup.poses.write_text(json.dumps({k: v for k, v in DEFAULT_POSES.items() if k != "above_tray"}))
    code, out = setup.run(capsys)                       # hand is the default
    assert code == 0 and "hand mode" in out.splitlines()[0]
    code, out = setup.run(capsys, "--to", "tray")
    assert code == 1 and "tray mode" in out.splitlines()[0] and "pose 'above_tray' is missing" in out
    code, out = setup.run(capsys, "--to", "hand")
    assert code == 0 and "hand mode" in out.splitlines()[0]


def test_a_mode_that_does_not_exist_is_refused():
    with pytest.raises(SystemExit) as exc:
        main(["--to", "floor"])
    assert exc.value.code == 2


# --- robot.yaml ------------------------------------------------------------------------

def test_a_missing_robot_yaml_fails(setup):
    setup.config.unlink()
    checks = setup.checks()
    failed = one(checks, "robot.yaml not found", FAIL)
    assert str(setup.config) in failed.text
    assert one(checks, "ports and calibration not checked", INFO)    # nothing to read them from
    assert find(checks, " exists") == [] and find(checks, "calibration found") == []
    assert one(checks, "poses file valid", OK)                  # the poses still get checked


def test_robot_yaml_that_does_not_parse_fails(setup):
    setup.config.write_text("arm: [unclosed\n")
    failed = one(setup.checks(), "robot.yaml arm settings", FAIL)
    assert "ParserError" in failed.text


@pytest.mark.parametrize("override, message", [
    ({"move_speed_deg_s": 60}, "above the 45 deg/s arm limit"),
    ({"move_speed_deg_s": 0}, "arm.move_speed_deg_s must be a positive number"),
    ({"max_relative_target_deg": "ten"}, "arm.max_relative_target_deg must be a positive number"),
    ({"leader_id": DROP}, "arm.leader_id is missing"),
    ({"follower_port": ""}, "arm.follower_port is missing"),
])
def test_invalid_arm_settings_fail_with_the_pick_loaders_message(setup, override, message):
    setup.write_config(**override)
    checks = setup.checks()
    assert message in one(checks, "robot.yaml arm settings", FAIL).text
    assert one(checks, "ports and calibration not checked", INFO)
    assert find(checks, " exists") == [] and find(checks, "calibration found") == []


@pytest.mark.parametrize("handoff, message", [
    ({"timeout_s": 0, "grip": "squeeze"}, "handoff.timeout_s must be a positive number"),
    ({"timeout_s": 60, "grip": "firm"}, "handoff.grip must be 'squeeze' or 'light'"),
    ({"timeout_s": 60}, "handoff.grip must be 'squeeze' or 'light'"),
])
def test_invalid_handoff_settings_fail_in_hand_mode_and_warn_in_tray_mode(setup, handoff, message):
    setup.write_config(handoff=handoff)
    failed = one(setup.checks("hand"), "robot.yaml handoff settings", FAIL)
    assert message in failed.text
    warned = one(setup.checks("tray"), "robot.yaml handoff settings", WARN)
    assert message in warned.text and "only --to hand uses them" in warned.text


def test_a_missing_handoff_section_fails_in_hand_mode(setup):
    data = yaml.safe_load(setup.config.read_text())
    del data["handoff"]
    setup.config.write_text(yaml.safe_dump(data))
    assert "'handoff:' section is missing" in one(setup.checks(), "handoff settings", FAIL).text


def test_the_light_grip_is_valid(setup):
    setup.write_config(handoff={"timeout_s": 30, "grip": "light"})
    assert "timeout 30 s, grip light" in one(setup.checks(), "handoff settings valid", OK).text


# --- ports -----------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["follower", "leader"])
def test_a_port_that_does_not_exist_fails_and_says_how_to_fix_it(setup, role):
    getattr(setup, f"{role}_port").unlink()
    failed = one(setup.checks(), f"{role} port", FAIL)
    assert str(getattr(setup, f"{role}_port")) in failed.text
    hint = everything(failed)
    assert "plug it in or rerun lerobot-find-port" in hint
    assert "macOS can renumber ports after replugging" in hint
    assert f"arm.{role}_port" in hint


def test_two_missing_ports_each_fail(setup):
    setup.follower_port.unlink()
    setup.leader_port.unlink()
    checks = setup.checks()
    assert statuses(find(checks, " port ")) == [FAIL, FAIL]


def test_existing_ports_pass(setup):
    checks = setup.checks()
    assert one(checks, "follower port", OK).text.endswith("exists")
    assert one(checks, "leader port", OK).text.endswith("exists")


def test_the_same_port_for_both_arms_fails(setup):
    setup.write_config(leader_port=str(setup.follower_port))
    checks = setup.checks()
    failed = one(checks, "are both", FAIL)
    assert "arm.follower_port and arm.leader_port" in failed.text
    assert "lerobot-find-port" in everything(failed)


# --- calibration files -----------------------------------------------------------------

def test_calibration_root_follows_lerobots_rules(monkeypatch, tmp_path):
    for name in ("HF_LEROBOT_CALIBRATION", "HF_LEROBOT_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert calibration_root() == tmp_path / "home" / ".cache" / "huggingface" / "lerobot" / "calibration"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert calibration_root() == tmp_path / "xdg" / "huggingface" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert calibration_root() == tmp_path / "hf" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))
    assert calibration_root() == tmp_path / "lerobot" / "calibration"
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "cal"))
    assert calibration_root() == tmp_path / "cal"


def test_calibration_files_are_named_the_way_lerobot_names_them(tmp_path):
    assert calibration_path(tmp_path, "follower", "arm_a") == tmp_path / "robots" / "so_follower" / "arm_a.json"
    assert calibration_path(tmp_path, "leader", "arm_b") == tmp_path / "teleoperators" / "so_leader" / "arm_b.json"


def test_the_installed_lerobot_names_calibration_files_the_same_way():
    """Guards against LeRobot moving its calibration folders (skipped where it isn't installed)."""
    constants = pytest.importorskip("lerobot.utils.constants")
    assert calibration_root() == constants.HF_LEROBOT_CALIBRATION
    assert (calibration_path(constants.HF_LEROBOT_CALIBRATION, "follower", "x")
            == constants.HF_LEROBOT_CALIBRATION / constants.ROBOTS / "so_follower" / "x.json")
    assert (calibration_path(constants.HF_LEROBOT_CALIBRATION, "leader", "x")
            == constants.HF_LEROBOT_CALIBRATION / constants.TELEOPERATORS / "so_leader" / "x.json")


def test_both_calibration_files_found_pass(setup):
    checks = setup.checks()
    assert str(setup.cal("follower")) in one(checks, "follower (so_follower robot) calibration found", OK).text
    assert str(setup.cal("leader")) in one(checks, "leader (so_leader teleoperator) calibration found", OK).text


def test_a_missing_follower_calibration_fails_with_the_calibrate_command(setup):
    setup.cal("follower").unlink()
    checks = setup.checks()
    failed = one(checks, "follower (so_follower robot) calibration not found", FAIL)
    assert str(setup.cal("follower")) in failed.text
    hint = everything(failed)
    assert "lerobot-calibrate --robot.type=so101_follower" in hint
    assert f"--robot.port={setup.follower_port}" in hint and f"--robot.id={FOLLOWER_ID}" in hint
    assert find(checks, "calibrated span") == [] and find(checks, "spans agree") == []     # nothing to compare


def test_a_missing_leader_calibration_fails_with_the_calibrate_command(setup):
    setup.cal("leader").unlink()
    failed = one(setup.checks(), "leader (so_leader teleoperator) calibration not found", FAIL)
    hint = everything(failed)
    assert "lerobot-calibrate --teleop.type=so101_leader" in hint
    assert f"--teleop.port={setup.leader_port}" in hint and f"--teleop.id={LEADER_ID}" in hint


def test_a_calibration_for_another_id_does_not_count(setup):
    setup.write_config(leader_id="so101_leader_arm_b")      # only so101_leader_arm.json exists
    failed = one(setup.checks(), "leader (so_leader teleoperator) calibration not found", FAIL)
    assert failed.text.endswith("so101_leader_arm_b.json")


def test_a_calibration_file_that_is_not_json_fails(setup):
    setup.cal("leader").write_text("{ not json")
    failed = one(setup.checks(), "leader (so_leader teleoperator) calibration:", FAIL)
    assert "can't be read" in failed.text


def test_a_calibration_file_missing_a_joint_range_fails(setup):
    data = json.loads(setup.cal("follower").read_text())
    del data["elbow_flex"]["range_max"]
    setup.cal("follower").write_text(json.dumps(data))
    failed = one(setup.checks(), "follower (so_follower robot) calibration:", FAIL)
    assert "no range_min and range_max for 'elbow_flex'" in failed.text


def test_a_calibration_file_with_an_empty_range_fails(setup):
    write_calibration(setup.cal("follower"), wrist_flex=0)
    assert "'wrist_flex' has range_max" in one(setup.checks(), "follower (so_follower robot) calibration:", FAIL).text


# --- calibration ranges ----------------------------------------------------------------

def test_matching_spans_pass_and_say_the_gripper_is_skipped(setup):
    ok = one(setup.checks(), "calibrated spans agree within 15%", OK)
    assert "gripper skipped" in ok.text


def test_the_wrist_flex_126_vs_177_degree_example_warns_with_both_spans(setup):
    write_calibration(setup.cal("follower"), wrist_flex=WRIST_177)
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126)
    checks = setup.checks()
    warned = one(checks, "wrist_flex calibrated span differs", WARN)
    assert "follower 177°" in warned.text and "leader 126°" in warned.text
    assert "29% apart" in warned.text
    assert find(checks, "spans agree") == []
    assert [c.status for c in checks if "calibrat" in c.text and c.status == FAIL] == []


def test_the_warning_suggests_backing_up_and_recalibrating_the_leader_with_a_full_sweep(setup):
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126)
    fix = one(setup.checks(), "recalibrate the leader", WARN)
    assert "wrist_flex" in fix.text
    steps = "\n".join(fix.hints)
    assert f"mkdir -p data/calibration && cp {setup.cal('leader')} data/calibration/{LEADER_ID}-" in steps
    assert f"lerobot-calibrate --teleop.type=so101_leader --teleop.port={setup.leader_port} " \
           f"--teleop.id={LEADER_ID}" in steps
    assert "sweep every joint fully" in steps
    assert steps.index("back up") < steps.index("recalibrate:")        # back up first


def test_a_shorter_follower_span_suggests_recalibrating_the_follower(setup):
    write_calibration(setup.cal("follower"), elbow_flex=1500)
    checks = setup.checks()
    assert "elbow_flex calibrated span differs: follower 132°, leader 195°" in one(checks, "elbow_flex calibrated span differs", WARN).text
    fix = one(checks, "recalibrate the follower", WARN)
    steps = "\n".join(fix.hints)
    assert f"cp {setup.cal('follower')} data/calibration/{FOLLOWER_ID}-" in steps
    assert f"lerobot-calibrate --robot.type=so101_follower --robot.port={setup.follower_port} " \
           f"--robot.id={FOLLOWER_ID}" in steps
    assert find(checks, "recalibrate the leader") == []


def test_each_differing_joint_warns_and_each_short_arm_gets_one_fix(setup):
    write_calibration(setup.cal("leader"), wrist_flex=WRIST_126, shoulder_lift=1500)
    write_calibration(setup.cal("follower"), elbow_flex=1500)
    checks = setup.checks()
    assert len(find(checks, "calibrated span differs", WARN)) == 3
    leader_fix = one(checks, "recalibrate the leader", WARN)
    assert "shoulder_lift, wrist_flex" in leader_fix.text
    assert "elbow_flex" in one(checks, "recalibrate the follower", WARN).text


def test_the_gripper_span_difference_is_ignored(setup):
    write_calibration(setup.cal("follower"), gripper=1298)
    write_calibration(setup.cal("leader"), gripper=700)         # almost half: normal for the gripper
    checks = setup.checks()
    assert find(checks, "gripper calibrated span") == []
    assert one(checks, "calibrated spans agree", OK)
    assert WARN not in statuses(checks)


@pytest.mark.parametrize("leader, warns", [
    (2000, False),
    (1700, False),          # exactly 15% of the larger span: not over
    (1699, True),
    (2400, True),           # either arm can be the shorter one
    (2300, False),
    (1850, False),
])
def test_spans_over_15_percent_apart_warn(setup, leader, warns):
    write_calibration(setup.cal("follower"), wrist_flex=2000)
    write_calibration(setup.cal("leader"), wrist_flex=leader)
    assert bool(find(setup.checks(), "wrist_flex calibrated span differs", WARN)) is warns


def test_spans_are_compared_by_range_not_by_where_the_range_sits(setup):
    data = json.loads(setup.cal("leader").read_text())
    data["wrist_flex"]["range_min"] += 700          # the same span, shifted along the servo's travel
    data["wrist_flex"]["range_max"] += 700
    setup.cal("leader").write_text(json.dumps(data))
    assert one(setup.checks(), "calibrated spans agree", OK)


# --- poses -----------------------------------------------------------------------------

def test_valid_poses_pass_in_hand_mode(setup):
    ok = one(setup.checks("hand"), "poses file valid for hand mode", OK)
    assert str(setup.poses) in ok.text and "present" in ok.text


def test_a_missing_poses_file_says_to_run_teach(setup):
    setup.poses.unlink()
    failed = one(setup.checks(), "poses file not found", FAIL)
    assert str(setup.poses) in failed.text
    assert "run python -m arm.teach" in failed.hints


def test_a_missing_poses_file_in_tray_mode_says_to_run_teach(setup):
    setup.poses.unlink()
    assert "run python -m arm.teach --with-tray" in one(setup.checks("tray"), "poses file not found", FAIL).hints


def test_poses_without_the_modes_pose_fail(setup):
    poses = {k: v for k, v in DEFAULT_POSES.items() if k != "present"}
    setup.poses.write_text(json.dumps(poses))
    failed = one(setup.checks("hand"), "poses file", FAIL)
    assert "pose 'present' is missing; hand mode needs it" in failed.text
    assert "re-teach with python -m arm.teach" in failed.hints[0]
    assert one(setup.checks("tray"), "poses file valid for tray mode", OK)       # tray doesn't need `present`


def test_poses_without_above_tray_pass_hand_mode_but_fail_tray_mode(setup):
    poses = {k: v for k, v in DEFAULT_POSES.items() if k != "above_tray"}
    setup.poses.write_text(json.dumps(poses))
    assert one(setup.checks("hand"), "poses file valid for hand mode", OK)
    failed = one(setup.checks("tray"), "poses file", FAIL)
    assert "pose 'above_tray' is missing; tray mode needs it" in failed.text
    assert "python -m arm.teach --with-tray" in failed.hints[0]


def test_a_suction_only_poses_file_passes_suction_mode_and_fails_the_claw_modes(setup):
    suction = {k: v for k, v in DEFAULT_POSES.items() if k == "ready" or k.startswith("suction_")}
    setup.poses.write_text(json.dumps(suction))
    ok = one(setup.checks("suction"), "poses file valid for suction mode", OK)
    assert "suction_press" in ok.text
    for mode in ("hand", "tray"):
        assert "is missing" in one(setup.checks(mode), "poses file", FAIL).text


def test_poses_without_the_suction_poses_fail_suction_mode_with_the_teach_hint(setup):
    poses = {k: v for k, v in DEFAULT_POSES.items() if not k.startswith("suction_")}
    setup.poses.write_text(json.dumps(poses))
    failed = one(setup.checks("suction"), "poses file", FAIL)
    assert "pose 'suction_above' is missing; suction mode needs it" in failed.text
    assert "re-teach with python -m arm.teach --grasp suction" in failed.hints[0]
    assert one(setup.checks("hand"), "poses file valid for hand mode", OK)


def test_the_mode_flag_accepts_suction_and_does_not_need_the_handoff_settings(setup, capsys):
    data = yaml.safe_load(setup.config.read_text())
    del data["handoff"]
    setup.config.write_text(yaml.safe_dump(data))
    code, out = setup.run(capsys, "--to", "suction")
    assert "suction mode" in out.splitlines()[0]
    assert code == 0, out                                   # a missing handoff section only warns


def test_poses_that_are_not_json_or_not_an_object_fail(setup):
    setup.poses.write_text("{ not json")
    assert one(setup.checks(), "poses file", FAIL)
    setup.poses.write_text("[1, 2, 3]")
    assert "top level must be an object" in one(setup.checks(), "poses file", FAIL).text


def test_poses_with_a_bad_gripper_range_fail(setup):
    setup.poses.write_text(json.dumps({**DEFAULT_POSES, "gripper_open": 5.0, "gripper_closed": 2.0}))
    assert "less than 10 apart" in one(setup.checks(), "poses file", FAIL).text


# --- cameras ---------------------------------------------------------------------------

def test_cameras_are_listed_with_their_indices_and_the_renumbering_reminder(setup):
    info = one(setup.checks(), "cameras in robot.yaml", INFO)
    assert "wrist=1, scene=2, mast_top=3" in info.text
    assert "not opened" in info.text
    assert "macOS can renumber cameras after replugging" in everything(info)


def test_a_camera_may_be_a_device_path(setup):
    setup.write_config(cameras={"wrist": "/dev/video0"})
    assert "wrist=/dev/video0" in one(setup.checks(), "cameras in robot.yaml", INFO).text


def test_camera_problems_only_warn(setup, capsys):
    setup.write_config(cameras={"wrist": None})
    assert "cameras.wrist must be an OpenCV index or device path" in one(setup.checks(), "cameras:", WARN).text
    setup.write_config(cameras={})
    assert "no cameras listed" in one(setup.checks(), "cameras:", WARN).text
    code, _ = setup.run(capsys)
    assert code == 0


# --- never touches hardware ------------------------------------------------------------

def test_the_check_opens_no_serial_port_camera_or_arm(setup):
    code = ("import sys\n"
            "import arm.check as c\n"
            "rc = c.main(['--config', sys.argv[1], '--poses', sys.argv[2]])\n"
            "loaded = [m for m in ('lerobot', 'serial', 'cv2', 'pyrealsense2', 'arm.real') if m in sys.modules]\n"
            "sys.exit(rc or (7 if loaded else 0))\n")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(setup.tmp), "HF_LEROBOT_CALIBRATION": str(setup.root)}
    proc = subprocess.run([sys.executable, "-c", code, str(setup.config), str(setup.poses)], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ports_are_only_checked_for_existence_never_opened(setup, monkeypatch):
    ports = {str(setup.follower_port), str(setup.leader_port)}
    opened = []
    real_open, real_os_open = open, check_mod.os.open

    def watching_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def watching_os_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", watching_open)
    monkeypatch.setattr(check_mod.os, "open", watching_os_open)
    assert set(statuses(setup.checks())) == {OK, INFO}
    assert ports.isdisjoint(opened)
    assert str(setup.config) in opened          # the watch works: the files it reads do show up
