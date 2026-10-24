"""Dry-run tests for teaching poses (task A3). No hardware."""

import json
import math
import tempfile
import time

import pytest
import yaml

import arm.real as real_mod
import arm.teach as teach_mod
from arm import pick, poses as poses_mod
from arm.sim import DEFAULT_POSES, SimArm, SimLeader
from arm.teach import STEPS, TeachError, dry_run_targets, save_with_backup, scripted_waiter, teach

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


def scripted(leader, follower, poses=DEFAULT_POSES):
    return scripted_waiter(leader, follower, dry_run_targets(poses))


def assert_matches_default(taught):
    for name in poses_mod.POSE_NAMES:
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
    # the taught poses drive a clean simulated pick
    assert pick.pick_to_tray(SimArm(reloaded), reloaded)["result"] == "in_tray"


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
    assert asked == ["ready", "above_object", "at_object", "at_object", "lift", "above_tray",
                     "gripper_open", "gripper_closed"]
    assert any("at_object is within" in m and "above_object" in m for m in logs)
    assert_matches_default(taught)


def test_claw_steps_may_repeat_the_tray_pose():
    """gripper_open / gripper_closed only change the claw, so they are not 'unmoved'."""
    poses = json.loads(json.dumps(DEFAULT_POSES))
    poses["above_tray"]["gripper"] = poses["gripper_open"]      # claw already open over the tray
    leader, follower = SimLeader(), SimArm()
    logs = []
    taught = teach(leader, follower, speed_deg_s=45, wait_for=scripted(leader, follower, poses),
                   log=logs.append)
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
        def __init__(self, port, robot_id, max_relative_target):
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


def test_cli_invalid_poses_are_not_saved(tmp_path, capsys, monkeypatch, fake_hardware, default_poses_path):
    """The claw open and closed steps taught at the same value: refused, nothing written."""
    before = default_poses_path.read_bytes()
    poses = json.loads(json.dumps(DEFAULT_POSES))
    poses["gripper_closed"] = poses["gripper_open"] - 3

    def person(step, prompt):
        if not hasattr(fake_hardware, "script"):
            fake_hardware.script = scripted(fake_hardware.leader, fake_hardware.follower, poses)
        return fake_hardware.script(step, prompt)

    monkeypatch.setattr(teach_mod, "enter_waiter", person)
    rc = teach_mod.main(["--config", str(write_config(tmp_path))])
    assert rc == 1 and "less than 10 apart" in capsys.readouterr().err
    assert default_poses_path.read_bytes() == before


def test_cli_help_needs_no_config(capsys):
    with pytest.raises(SystemExit) as exc:
        teach_mod.main(["--help"])
    assert exc.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
