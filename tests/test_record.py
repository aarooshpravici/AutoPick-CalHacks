"""Recording command tests (task T1). The command is checked with lerobot-record's own
config parser; nothing is recorded, nothing is exec'd and no hardware is opened."""

import copy
import shlex

import pytest
import yaml

from training import record

CFG = {
    "arm": {"follower_port": "/dev/tty.usbmodemF", "leader_port": "/dev/tty.usbmodemL",
            "follower_id": "so101_follower_arm", "leader_id": "so101_leader_arm",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
}
FAKE_EXE = "/fake/env/bin/lerobot-record"


class Execd(Exception):
    """Raised by the fake os.execv, which (like the real one) must not return."""


def build(cfg=CFG, **kw):
    args = dict(name="rehearsal", episodes=5, cameras=["wrist"], width=320, height=180, fps=30,
                episode_time_s=20, reset_time_s=10, resume=False, exe="lerobot-record")
    args.update(kw)
    return record.build_command(cfg, **args)


def parse_with_lerobot(cmd):
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    return draccus.parse(RecordConfig, args=cmd[1:])


def write_cfg(tmp_path, cfg=CFG):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A --run environment with every side effect faked: datasets under tmp, a fake
    lerobot-record path, an interactive terminal and an execv that records its call."""
    monkeypatch.setattr(record, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(record, "find_lerobot_record", lambda: FAKE_EXE)
    monkeypatch.setattr(record, "interactive", lambda: True)
    calls = {"execv": [], "prompts": []}

    def fake_execv(path, argv):
        calls["execv"].append((path, argv))
        raise Execd

    def fake_input(prompt=""):
        calls["prompts"].append(prompt)
        return calls.get("answer", "")

    monkeypatch.setattr(record.os, "execv", fake_execv)
    monkeypatch.setattr("builtins.input", fake_input)
    calls["config"] = str(write_cfg(tmp_path))
    return calls


def run_live(live, *extra, name="demo1"):
    return record.main(["--name", name, "--episodes", "3", "--config", live["config"], "--run", *extra])


# --- the command itself ---------------------------------------------------------------

def test_lerobot_accepts_the_command():
    cfg = parse_with_lerobot(build())
    assert cfg.robot.port == "/dev/tty.usbmodemF" and cfg.robot.id == "so101_follower_arm"
    assert cfg.teleop.port == "/dev/tty.usbmodemL" and cfg.teleop.id == "so101_leader_arm"
    cam = cfg.robot.cameras["wrist"]
    assert (cam.index_or_path, cam.width, cam.height, cam.fps) == (1, 320, 180, 30)
    assert cfg.dataset.num_episodes == 5 and cfg.dataset.push_to_hub is False
    assert str(cfg.dataset.root).endswith("data/datasets/rehearsal")


def test_scene_camera_keeps_its_name():
    cfg = parse_with_lerobot(build(cameras=["wrist", "scene"]))
    assert set(cfg.robot.cameras) == {"wrist", "scene"}
    assert cfg.robot.cameras["scene"].index_or_path == 2


def test_missing_camera_index():
    with pytest.raises(SystemExit, match="scene"):
        record.camera_spec(["wrist", "scene"], {"wrist": 1}, 320, 180, 30)


# --- speed: robot.yaml is the ceiling, applied as LeRobot's per-command step ----------

def test_default_speed_is_the_robot_yaml_ceiling():
    cfg = parse_with_lerobot(build())
    assert isinstance(cfg.robot.max_relative_target, float)
    assert cfg.robot.max_relative_target == pytest.approx(45 / 30, abs=1e-3)


def test_lower_speed_lowers_the_step():
    cfg = parse_with_lerobot(build(speed_deg_s=25))
    assert cfg.robot.max_relative_target == pytest.approx(25 / 30, abs=1e-3)


def test_step_never_exceeds_max_relative_target_deg():
    fast = copy.deepcopy(CFG)
    fast["arm"]["move_speed_deg_s"] = 600  # 600 / 30 = 20 per command > the 10 cap
    assert parse_with_lerobot(build(cfg=fast)).robot.max_relative_target == pytest.approx(10.0)


def test_speed_above_ceiling_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit, match="above robot.yaml arm.move_speed_deg_s"):
        build(speed_deg_s=60)
    with pytest.raises(SystemExit, match="above"):
        record.main(["--name", "x", "--config", str(write_cfg(tmp_path)), "--speed", "46"])
    assert "lerobot-record" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
def test_non_positive_speed_is_refused(bad):
    with pytest.raises(SystemExit, match="--speed"):
        build(speed_deg_s=bad)


# --- robot.yaml problems are clear errors, with no hard-coded fallbacks ---------------

@pytest.mark.parametrize("key", ["max_relative_target_deg", "move_speed_deg_s", "follower_port",
                                 "leader_port", "follower_id", "leader_id"])
def test_missing_arm_key_names_it(key):
    cfg = copy.deepcopy(CFG)
    del cfg["arm"][key]
    with pytest.raises(SystemExit, match=f"arm.{key}"):
        build(cfg=cfg)


def test_missing_arm_section():
    with pytest.raises(SystemExit, match="arm.move_speed_deg_s"):
        build(cfg={"cameras": CFG["cameras"]})


def test_missing_cameras_section():
    with pytest.raises(SystemExit, match="cameras"):
        build(cfg={"arm": CFG["arm"]})


def test_non_numeric_safety_limit_is_refused():
    cfg = copy.deepcopy(CFG)
    cfg["arm"]["max_relative_target_deg"] = "ten"
    with pytest.raises(SystemExit, match="max_relative_target_deg must be a positive number"):
        build(cfg=cfg)


def test_missing_config_file_points_at_s6(tmp_path):
    with pytest.raises(SystemExit, match="not found.*S6"):
        record.main(["--name", "x", "--config", str(tmp_path / "nope.yaml")])


def test_empty_config_file(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text("")
    with pytest.raises(SystemExit, match="empty"):
        record.main(["--name", "x", "--config", str(path)])


# --- default mode: print only, never exec ---------------------------------------------

def test_default_prints_and_never_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(record.os, "execv", lambda *a: pytest.fail("execv without --run"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted without --run"))
    assert record.main(["--name", "demo1", "--episodes", "3", "--config", str(write_cfg(tmp_path))]) == 0
    out, err = capsys.readouterr()
    assert "--dataset.num_episodes=3" in out and "local/demo1" in out
    assert "--run" in err


def test_printed_command_round_trips_through_the_shell(tmp_path, capsys):
    record.main(["--name", "demo1", "--scene", "--config", str(write_cfg(tmp_path)), "--print-only"])
    cmd = shlex.split(capsys.readouterr().out)
    cfg = parse_with_lerobot(cmd)
    assert set(cfg.robot.cameras) == {"wrist", "scene"}
    assert cfg.dataset.single_task == record.TASK


def test_dry_run_alias(tmp_path, capsys):
    assert record.main(["--name", "demo1", "--config", str(write_cfg(tmp_path)), "--dry-run"]) == 0
    assert "local/demo1" in capsys.readouterr().out


def test_run_and_print_only_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        record.main(["--name", "x", "--config", str(write_cfg(tmp_path)), "--run", "--print-only"])


def test_bad_name():
    with pytest.raises(SystemExit):
        record.main(["--name", "../x"])


@pytest.mark.parametrize("flag,value", [("--episodes", "0"), ("--fps", "0"), ("--reset-time", "-1")])
def test_bad_counts(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        record.main(["--name", "x", "--config", str(write_cfg(tmp_path)), flag, value])


# --- --run: confirmation, then exec of the full path ----------------------------------

def test_run_execs_full_path_after_enter(live, capsys):
    with pytest.raises(Execd):
        run_live(live, "--speed", "25")
    assert len(live["prompts"]) == 1 and "clear" in live["prompts"][0].lower()
    [(path, argv)] = live["execv"]
    assert path == FAKE_EXE and argv[0] == FAKE_EXE
    assert "--robot.max_relative_target=0.833" in argv
    assert "25 deg/s" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["n", "no", " x "])
def test_run_cancelled_by_any_other_answer(live, answer):
    live["answer"] = answer
    with pytest.raises(SystemExit, match="cancelled"):
        run_live(live)
    assert live["execv"] == []


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_run_cancelled_by_eof_or_ctrl_c(live, monkeypatch, exc):
    def raise_(*a):
        raise exc
    monkeypatch.setattr("builtins.input", raise_)
    with pytest.raises(SystemExit, match="cancelled"):
        run_live(live)
    assert live["execv"] == []


def test_run_refused_without_a_terminal(live, monkeypatch):
    monkeypatch.setattr(record, "interactive", lambda: False)
    with pytest.raises(SystemExit, match="interactive terminal"):
        run_live(live)
    assert live["execv"] == [] and live["prompts"] == []


def test_run_refused_when_lerobot_record_is_missing(live, monkeypatch):
    monkeypatch.setattr(record, "find_lerobot_record", lambda: None)
    with pytest.raises(SystemExit, match="lerobot-record not found"):
        run_live(live)
    assert live["execv"] == [] and live["prompts"] == []


def test_find_lerobot_record_falls_back_to_path(monkeypatch):
    seen = []

    def fake_which(name, path=None):
        seen.append(path)
        return None if path else "/on/path/lerobot-record"
    monkeypatch.setattr(record.shutil, "which", fake_which)
    assert record.find_lerobot_record() == "/on/path/lerobot-record"
    assert seen[0] is not None and seen[1] is None


def test_run_refuses_to_overwrite_an_existing_dataset(live):
    (record.DATASETS / "demo1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="already exists"):
        run_live(live)
    assert live["execv"] == [] and live["prompts"] == []


def test_resume_of_missing_dataset_is_refused(live):
    with pytest.raises(SystemExit, match="nothing to resume"):
        run_live(live, "--resume")
    assert live["execv"] == [] and live["prompts"] == []
    assert not (record.DATASETS / "demo1").exists()


def test_resume_of_existing_dataset_runs(live):
    meta = record.DATASETS / "demo1" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    with pytest.raises(Execd):
        run_live(live, "--resume")
    [(_, argv)] = live["execv"]
    assert "--resume=true" in argv
