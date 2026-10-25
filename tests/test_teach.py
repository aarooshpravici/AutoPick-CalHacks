"""Dry-run tests for teaching poses (task A3). No hardware."""

import json
import math
import tempfile
import time

import pytest
import yaml

import arm.real as real_mod
from arm import align as align_mod
import arm.teach as teach_mod
from arm import pick, poses as poses_mod
from arm.sim import DEFAULT_POSES, SimArm, SimLeader
from arm.teach import (STEPS, TeachError, dry_run_targets, save_with_backup, scripted_waiter, teach,
                       teach_steps)

QUIET = lambda msg: None  # noqa: E731

# Same keys as config/robot.yaml (task S6); the real file is never read by the tests.
ROBOT_YAML = {
    "arm": {
        "follower_port": "/dev/tty.usbmodemTEST1",
        "leader_port": "/dev/tty.usbmodemTEST2",
        "follower_id": "so101_follower_arm",
        "leader_id": "so101_leader_arm",
        "max_relative_target_deg": 10,
        "move_speed_deg_s": 45,
    },
}
DROP = object()     # write_config(...=DROP) leaves the key out


def write_config(tmp_path, **arm_overrides):
    data = json.loads(json.dumps(ROBOT_YAML))
    for key, value in arm_overrides.items():
        if value is DROP:
            data["arm"].pop(key, None)
        else:
            data["arm"][key] = value
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def scripted(leader, follower, poses=DEFAULT_POSES, with_tray=False):
    return scripted_waiter(leader, follower, dry_run_targets(poses, with_tray))


def assert_matches_default(taught, with_tray=False):
    names = [name for name in poses_mod.POSE_NAMES if name in taught]
    assert names == ["ready", "above_object", "at_object", "lift", "present"] + (["above_tray"] if with_tray else [])
    for name in names:
        for j in poses_mod.JOINTS:
            assert abs(taught[name][j] - DEFAULT_POSES[name][j]) < 0.5, (name, j)
    assert abs(taught["gripper_open"] - DEFAULT_POSES["gripper_open"]) < 0.5
    assert abs(taught["gripper_closed"] - DEFAULT_POSES["gripper_closed"]) < 0.5


@pytest.fixture
def default_poses_path(tmp_path, monkeypatch):
    """Point config/poses.json at a temp file holding a sentinel 'real' taught pose set."""
    path = tmp_path / "config" / "poses.json"
    real = json.loads(json.dumps(DEFAULT_POSES))
    real["ready"]["shoulder_pan"] = 12.5            # marks it as the person's taught file
    poses_mod.save(real, path)
    monkeypatch.setattr(poses_mod, "DEFAULT_POSES_PATH", path)
    monkeypatch.setattr(teach_mod, "REPO_ROOT", tmp_path)     # backups go under tmp_path/data
    return path


@pytest.fixture
def scratch_tmp(tmp_path, monkeypatch):
    """tempfile.mkdtemp() lands under tmp_path/scratch."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


# --- teach() -------------------------------------------------------------------

def test_teach_records_every_pose_and_saves(tmp_path):
    leader, follower = SimLeader(), SimArm()
    taught = teach(leader, follower, speed_deg_s=45, wait_for=scripted(leader, follower), log=QUIET)
    assert_matches_default(taught)

    out = tmp_path / "poses.json"
    save_with_backup(taught, out)
    reloaded = poses_mod.load(out)
    assert reloaded["at_object"] == taught["at_object"]
    # the taught poses drive a clean simulated pick and hand-off
    assert pick.pick_to_hand(SimArm(reloaded), reloaded)["result"] == "taken"


# --- A7: present is taught after lift; above_tray only with --with-tray ----------

def test_steps_teach_present_right_after_lift_and_no_tray():
    names = [name for name, _ in STEPS]
    assert names == ["ready", "above_object", "at_object", "lift", "present", "gripper_open", "gripper_closed"]
    assert [name for name, _ in teach_steps(with_tray=True)] == [
        "ready", "above_object", "at_object", "lift", "present", "above_tray", "gripper_open", "gripper_closed"]


def test_teach_without_tray_is_refused_by_tray_mode(tmp_path):
    leader, follower = SimLeader(), SimArm()
    taught = teach(leader, follower, speed_deg_s=45, wait_for=scripted(leader, follower), log=QUIET)
    assert "present" in taught and "above_tray" not in taught
    poses_mod.save(taught, tmp_path / "poses.json")
    reloaded = poses_mod.load(tmp_path / "poses.json")
    assert "above_tray" not in reloaded
    with pytest.raises(poses_mod.PosesError, match="above_tray.*--with-tray"):
        pick.pick_to_tray(SimArm(reloaded), reloaded)


def test_teach_with_tray_records_above_tray_too(tmp_path):
    leader, follower = SimLeader(), SimArm()
    asked = []
    script = scripted(leader, follower, with_tray=True)
    taught = teach(leader, follower, speed_deg_s=45, log=QUIET, with_tray=True,
                   wait_for=lambda step, prompt: asked.append(step) or script(step, prompt))
    assert asked == [name for name, _ in teach_steps(with_tray=True)]
    assert_matches_default(taught, with_tray=True)
    poses_mod.save(taught, tmp_path / "poses.json")
    reloaded = poses_mod.load(tmp_path / "poses.json")
    assert pick.pick_to_tray(SimArm(reloaded), reloaded)["result"] == "in_tray"
    assert pick.pick_to_hand(SimArm(reloaded), reloaded)["result"] == "taken"


def test_cli_dry_run_with_tray(tmp_path, capsys, default_poses_path, scratch_tmp):
    before = default_poses_path.read_bytes()
    rc = teach_mod.main(["--dry-run", "--with-tray", "--config", str(write_config(tmp_path))])
    assert rc == 0, capsys.readouterr()
    assert default_poses_path.read_bytes() == before            # config/poses.json untouched
    written = list(scratch_tmp.glob("arm-teach-dry-run-*/poses.json"))
    assert len(written) == 1
    assert_matches_default(poses_mod.load(written[0]), with_tray=True)


def test_teach_follower_is_speed_capped():
    leader, follower = SimLeader(), SimArm()
    sent = []
    command = follower.command
    follower.command = lambda t: (sent.append(dict(t)), command(t))
    teach(leader, follower, speed_deg_s=25, wait_for=scripted(leader, follower), log=QUIET)
    assert len(sent) > 100
    for a, b in zip(sent, sent[1:]):
        for j in poses_mod.ARM_JOINTS:
            assert abs(b[j] - a[j]) <= 25 / pick.RATE_HZ + 1e-6


# review: teach speed was unvalidated (negative drove away from the leader)
@pytest.mark.parametrize("speed", [0, -45, 46, 90, math.nan, math.inf, None, True, "25"])
def test_teach_refuses_bad_speed_before_moving(speed):
    leader, follower = SimLeader(dict(DEFAULT_POSES["lift"])), SimArm()
    sent = []
    follower.command = lambda t: sent.append(t)
    with pytest.raises(ValueError, match="speed_deg_s"):
        teach(leader, follower, speed_deg_s=speed, wait_for=scripted(leader, follower), log=QUIET)
    assert sent == []


def test_existing_poses_are_backed_up(tmp_path, monkeypatch):
    monkeypatch.setattr(teach_mod, "REPO_ROOT", tmp_path)
    out = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, out)
    backup = save_with_backup(DEFAULT_POSES, out)
    assert backup is not None and backup.exists()


# review: an extra Enter recorded the next pose unmoved, as a copy of the previous one
def test_unmoved_pose_is_not_recorded_and_the_step_is_asked_again():
    leader, follower = SimLeader(), SimArm()
    script = scripted(leader, follower)
    asked = []

    class PressedAlready:
        def is_set(self):
            return True

    def wait_for(step, prompt):
        asked.append(step)
        if step == "at_object" and asked.count("at_object") == 1:
            return PressedAlready()     # double Enter: the leader is still at above_object
        return script(step, prompt)

    logs = []
    taught = teach(leader, follower, speed_deg_s=45, wait_for=wait_for, log=logs.append)
    assert asked == ["ready", "above_object", "at_object", "at_object", "lift", "present",
                     "gripper_open", "gripper_closed"]
    assert any("at_object is within" in m and "above_object" in m for m in logs)
    assert_matches_default(taught)


@pytest.mark.parametrize("with_tray, last", [(False, "present"), (True, "above_tray")])
def test_claw_steps_may_repeat_the_last_pose(with_tray, last):
    """gripper_open / gripper_closed only change the claw, so they are not 'unmoved'."""
    poses = json.loads(json.dumps(DEFAULT_POSES))
    poses[last]["gripper"] = poses["gripper_open"]      # claw already open at the last pose
    leader, follower = SimLeader(), SimArm()
    logs = []
    taught = teach(leader, follower, speed_deg_s=45, wait_for=scripted(leader, follower, poses, with_tray),
                   log=logs.append, with_tray=with_tray)
    assert not any("within" in m for m in logs)
    assert abs(taught["gripper_open"] - poses["gripper_open"]) < 0.5


def test_scripted_waiter_gives_up_when_the_follower_never_arrives():
    leader, follower = SimLeader(), SimArm(stall_joint="elbow_flex")
    wait_for = scripted_waiter(leader, follower, dry_run_targets(DEFAULT_POSES), limit_s=5)
    with pytest.raises(TeachError, match="never reached above_object"):
        teach(leader, follower, speed_deg_s=45, wait_for=wait_for, log=QUIET)


class FakeStdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty

    def fileno(self):
        return 0


@pytest.mark.parametrize("tty", [True, False])
def test_enter_waiter_flushes_typed_ahead_input(monkeypatch, tty):
    import termios
    events = []
    monkeypatch.setattr(teach_mod.sys, "stdin", FakeStdin(tty))
    monkeypatch.setattr(termios, "tcflush", lambda fd, how: events.append(("flush", how)))
    monkeypatch.setattr("builtins.input", lambda prompt: events.append(("input", prompt)) or "")
    pressed = teach_mod.enter_waiter("ready", "[ready] go: ")
    deadline = time.monotonic() + 5
    while not pressed.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    expected = [("flush", termios.TCIFLUSH)] if tty else []
    assert events == expected + [("input", "[ready] go: ")]


def test_enter_waiter_reports_closed_input(monkeypatch):
    def eof(prompt):
        raise EOFError
    monkeypatch.setattr(teach_mod.sys, "stdin", FakeStdin(False))
    monkeypatch.setattr("builtins.input", eof)
    pressed = teach_mod.enter_waiter("ready", "")
    deadline = time.monotonic() + 5
    with pytest.raises(TeachError, match="input closed"):
        while True:
            assert time.monotonic() < deadline
            pressed.is_set()
            time.sleep(0.01)


# --- command line: dry run (review: --dry-run could never succeed) --------------

def test_cli_dry_run_succeeds_and_writes_only_to_a_scratch_folder(
        tmp_path, capsys, default_poses_path, scratch_tmp):
    before = default_poses_path.read_bytes()
    rc = teach_mod.main(["--dry-run", "--config", str(write_config(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert default_poses_path.read_bytes() == before            # config/poses.json untouched
    written = list(scratch_tmp.glob("arm-teach-dry-run-*/poses.json"))
    assert len(written) == 1 and f"saved and reloaded {written[0]}" in out
    assert_matches_default(poses_mod.load(written[0]))
    assert not (tmp_path / "data").exists()                     # nothing to back up


def test_cli_dry_run_writes_an_explicit_out(tmp_path, capsys, default_poses_path, scratch_tmp):
    target = tmp_path / "taught" / "poses.json"
    rc = teach_mod.main(["--dry-run", "--out", str(target), "--config", str(write_config(tmp_path))])
    assert rc == 0, capsys.readouterr()
    assert_matches_default(poses_mod.load(target))
    assert list(scratch_tmp.iterdir()) == []


@pytest.mark.parametrize("relative", [False, True])
def test_cli_dry_run_refuses_to_write_config_poses(tmp_path, capsys, monkeypatch, default_poses_path,
                                                   relative):
    before = default_poses_path.read_bytes()
    out = default_poses_path
    if relative:
        monkeypatch.chdir(default_poses_path.parent)
        out = "../config/poses.json"
    rc = teach_mod.main(["--dry-run", "--out", str(out), "--config", str(write_config(tmp_path))])
    assert rc == 1
    assert "never writes" in capsys.readouterr().err
    assert default_poses_path.read_bytes() == before
    assert not (tmp_path / "data").exists()


# --- command line: speed and robot.yaml (review: speed unvalidated) -------------

@pytest.fixture
def seen_speed(monkeypatch):
    seen = []
    real_teach = teach_mod.teach

    def spy(leader, follower, speed_deg_s, **kwargs):
        seen.append(speed_deg_s)
        return real_teach(leader, follower, speed_deg_s, **kwargs)

    monkeypatch.setattr(teach_mod, "teach", spy)
    return seen


@pytest.mark.parametrize("ceiling, extra, expected", [
    (25, [], 25), (45, [], 45), (45, ["--speed", "25"], 25), (25, ["--speed", "25"], 25),
])
def test_cli_speed_defaults_to_and_stays_under_config(tmp_path, capsys, scratch_tmp, seen_speed,
                                                     ceiling, extra, expected):
    rc = teach_mod.main(["--dry-run", *extra, "--config",
                         str(write_config(tmp_path, move_speed_deg_s=ceiling))])
    assert rc == 0, capsys.readouterr()
    assert seen_speed == [expected]


@pytest.mark.parametrize("speed, words", [
    ("30", ["30", "move_speed_deg_s", "25"]),     # above the robot.yaml ceiling
    ("0", ["positive"]), ("-10", ["positive"]), ("nan", ["positive"]), ("inf", ["positive"]),
])
def test_cli_refuses_bad_speed(tmp_path, capsys, scratch_tmp, seen_speed, speed, words):
    rc = teach_mod.main(["--dry-run", f"--speed={speed}", "--config",
                         str(write_config(tmp_path, move_speed_deg_s=25))])
    assert rc == 1
    err = capsys.readouterr().err
    assert all(w in err for w in words), err
    assert seen_speed == []
    assert list(scratch_tmp.iterdir()) == []


@pytest.mark.parametrize("overrides, words", [
    ({"move_speed_deg_s": 90}, ["move_speed_deg_s", "90"]),
    ({"move_speed_deg_s": DROP}, ["move_speed_deg_s"]),
    ({"move_speed_deg_s": -5}, ["move_speed_deg_s"]),
    ({"max_relative_target_deg": DROP}, ["max_relative_target_deg"]),
    ({"leader_id": DROP}, ["leader_id"]),
    ({"leader_port": ""}, ["leader_port"]),
])
@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_bad_config_is_a_clear_error(tmp_path, capsys, fake_hardware, seen_speed, overrides, words,
                                         mode):
    extra = ["--dry-run"] if mode == "dry" else []
    rc = teach_mod.main([*extra, "--config", str(write_config(tmp_path, **overrides))])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and all(w in err for w in words), err
    assert "Traceback" not in err
    assert seen_speed == [] and fake_hardware.opened == []


def test_cli_missing_config_has_no_fallback(tmp_path, capsys, seen_speed):
    rc = teach_mod.main(["--dry-run", "--config", str(tmp_path / "none.yaml")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
    assert seen_speed == []


# --- command line: real mode against fake hardware (review: unsafe open/close) ---

class Hardware:
    """Records what the fake real arms do. The fakes are simulated arms underneath."""

    def __init__(self):
        self.opened = []
        self.closed = []
        self.fail_open = set()
        self.fail_close = set()


@pytest.fixture
def fake_hardware(monkeypatch):
    hw = Hardware()

    class FakeRealLeader(SimLeader):
        def __init__(self, port, robot_id):
            if "leader" in hw.fail_open:
                raise RuntimeError("no matching calibration for the leader")
            super().__init__()
            hw.opened.append(("leader", port, robot_id))
            hw.leader = self

        def close(self):
            hw.closed.append("leader")
            if "leader" in hw.fail_close:
                raise OSError("port vanished")

    class FakeRealArm(SimArm):
        def __init__(self, port, robot_id, max_relative_target, frozen=()):
            if "follower" in hw.fail_open:
                raise RuntimeError("no matching calibration for the follower")
            super().__init__()
            hw.opened.append(("follower", port, robot_id, max_relative_target))
            hw.follower = self

        def close(self):
            hw.closed.append("follower")
            if "follower" in hw.fail_close:
                raise OSError("port vanished")

    def person(step, prompt):          # stands in for enter_waiter: no stdin
        hw.prompts = getattr(hw, "prompts", []) + [prompt]
        if not hasattr(hw, "script"):
            hw.script = scripted(hw.leader, hw.follower)
        return hw.script(step, prompt)

    monkeypatch.setattr(real_mod, "RealLeader", FakeRealLeader)
    monkeypatch.setattr(real_mod, "RealArm", FakeRealArm)
    monkeypatch.setattr(teach_mod, "enter_waiter", person)
    # The person also lines the leader up with the follower before teaching starts (arm/align.py).
    monkeypatch.setattr(align_mod, "live_align", align_mod.scripted_align)
    return hw


def test_cli_real_mode_reads_ports_ids_and_limits_from_config(tmp_path, capsys, fake_hardware,
                                                              seen_speed, default_poses_path):
    rc = teach_mod.main(["--speed", "25", "--config",
                         str(write_config(tmp_path, max_relative_target_deg=7))])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert fake_hardware.opened == [
        ("leader", "/dev/tty.usbmodemTEST2", "so101_leader_arm"),
        ("follower", "/dev/tty.usbmodemTEST1", "so101_follower_arm", 7),
    ]
    assert seen_speed == [25]
    assert len(fake_hardware.prompts) == len(STEPS)
    assert sorted(fake_hardware.closed) == ["follower", "leader"]
    # real mode saves config/poses.json, backing up the old one
    assert_matches_default(poses_mod.load(default_poses_path))
    assert len(list((tmp_path / "data" / "poses_backup").glob("poses-*.json"))) == 1


def test_cli_leader_failure_never_opens_the_follower(tmp_path, capsys, fake_hardware, default_poses_path):
    before = default_poses_path.read_bytes()
    fake_hardware.fail_open.add("leader")
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    err = capsys.readouterr().err
    assert rc == 1 and "no matching calibration for the leader" in err and "Traceback" not in err
    assert fake_hardware.opened == [] and fake_hardware.closed == []
    assert default_poses_path.read_bytes() == before


def test_cli_follower_failure_closes_the_leader(tmp_path, capsys, fake_hardware, default_poses_path):
    fake_hardware.fail_open.add("follower")
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    assert rc == 1 and "no matching calibration for the follower" in capsys.readouterr().err
    assert fake_hardware.closed == ["leader"]


def test_cli_failing_follower_close_still_closes_the_leader(tmp_path, capsys, fake_hardware,
                                                            default_poses_path):
    fake_hardware.fail_close.add("follower")
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    err = capsys.readouterr().err
    assert rc == 0                                       # the poses were taught; still saved
    assert fake_hardware.closed == ["follower", "leader"]
    assert "closing FakeRealArm failed" in err and "port vanished" in err


def test_cli_interrupt_while_teaching_closes_both_and_saves_nothing(tmp_path, capsys, monkeypatch,
                                                                    fake_hardware, default_poses_path):
    before = default_poses_path.read_bytes()

    def interrupted(step, prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(teach_mod, "enter_waiter", interrupted)
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    assert rc == 1 and "nothing saved" in capsys.readouterr().err
    assert sorted(fake_hardware.closed) == ["follower", "leader"]
    assert default_poses_path.read_bytes() == before


def test_cli_bad_claw_steps_are_asked_again_and_the_session_is_kept(tmp_path, capsys, monkeypatch,
                                                                    fake_hardware, default_poses_path):
    """The claw open step taught nearly closed: both claw steps are asked again, nothing is
    thrown away, and the good second try is saved."""
    bad = json.loads(json.dumps(DEFAULT_POSES))
    bad["gripper_open"] = bad["gripper_closed"] + 3
    asked = []

    def person(step, prompt):
        asked.append(step)
        if not hasattr(fake_hardware, "first"):
            fake_hardware.first = scripted(fake_hardware.leader, fake_hardware.follower, bad)
            fake_hardware.again = scripted(fake_hardware.leader, fake_hardware.follower)
        return (fake_hardware.again if asked.count(step) > 1 else fake_hardware.first)(step, prompt)

    monkeypatch.setattr(teach_mod, "enter_waiter", person)
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0 and "Teach both claw steps again" in out
    assert asked[-4:] == ["gripper_open", "gripper_closed", "gripper_open", "gripper_closed"]
    assert asked.count("ready") == 1                # the arm poses weren't taught again
    assert_matches_default(poses_mod.load(default_poses_path))


def test_a_claw_gap_too_small_asks_both_claw_steps_again():
    leader, follower = SimLeader(), SimArm()
    bad = json.loads(json.dumps(DEFAULT_POSES))
    bad["gripper_closed"] = bad["gripper_open"] - 4     # "closed" taught barely closed
    first, again = scripted(leader, follower, bad), scripted(leader, follower)
    asked, logs = [], []

    def wait_for(step, prompt):
        asked.append(step)
        return (again if asked.count(step) > 1 else first)(step, prompt)

    taught = teach(leader, follower, speed_deg_s=45, wait_for=wait_for, log=logs.append)
    assert asked == ["ready", "above_object", "at_object", "lift", "present",
                     "gripper_open", "gripper_closed", "gripper_open", "gripper_closed"]
    assert any("less than 10 apart" in m for m in logs)
    assert_matches_default(taught)


def clamp_warning(**joints):
    """LeRobot's step-cap warning, worded as lerobot/robots/utils.py logs it."""
    from pprint import pformat
    return ("Relative goal position magnitude had to be clamped to be safe.\n"
            + pformat({j: {"original goal_pos": v, "safe goal_pos": v / 2} for j, v in joints.items()}, indent=4))


def test_claw_clamp_warning_is_muted_only_while_teaching(caplog):
    lerobot_log = __import__("logging").getLogger(teach_mod.CLAMP_LOGGER)

    class Warns(SimArm):
        def command(self, targets):
            lerobot_log.warning(clamp_warning(gripper=95.0))
            lerobot_log.warning(clamp_warning(elbow_flex=40.0))
            super().command(targets)

    leader, follower = SimLeader(), Warns()
    with caplog.at_level("WARNING", logger=teach_mod.CLAMP_LOGGER):
        teach(leader, follower, speed_deg_s=45, wait_for=scripted(leader, follower), log=QUIET)
        during = [r.getMessage() for r in caplog.records]
        caplog.clear()
        lerobot_log.warning(clamp_warning(gripper=95.0))
    assert during and all("'elbow_flex'" in m for m in during)     # an arm joint clamped still shows
    assert len(caplog.records) == 1                                 # muted only while teaching
    assert lerobot_log.filters == []


@pytest.mark.parametrize("message, shown", [
    (clamp_warning(gripper=95.0), False),
    (clamp_warning(gripper=95.0, wrist_flex=30.0), True),
    (clamp_warning(shoulder_lift=30.0), True),
    ("some other LeRobot warning about 'gripper'", True)])
def test_quiet_claw_clamp_filter(message, shown):
    import logging
    record = logging.LogRecord(teach_mod.CLAMP_LOGGER, logging.WARNING, __file__, 1, message, None, None)
    assert teach_mod._QuietClawClamp().filter(record) is shown


def test_cli_help_needs_no_config(capsys):
    with pytest.raises(SystemExit) as exc:
        teach_mod.main(["--help"])
    assert exc.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
