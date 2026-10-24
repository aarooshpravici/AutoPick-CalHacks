"""Recording tests (task T1). The lerobot-record command is checked with lerobot-record's own
config parser and never run. The native recorder (--run) runs against fakes only: no serial
port or camera is ever opened (the no_hardware fixture fails the test if one is), nothing is
exec'd, and every dataset is written under tmp_path."""

import contextlib
import copy
import json
import math
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import arm.real as real_mod
from arm import align as align_mod
from arm import policy as policy_mod
from arm.poses import JOINTS
from arm.real import FROZEN_READING
from arm.sim import DEFAULT_POSES
from training import record

CFG = {
    "arm": {"follower_port": "/dev/tty.usbmodemF", "leader_port": "/dev/tty.usbmodemL",
            "follower_id": "so101_follower_arm", "leader_id": "so101_leader_arm",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
}
FROZEN_CFG = copy.deepcopy(CFG)
FROZEN_CFG["arm"]["frozen_joints"] = ["wrist_roll"]


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


@pytest.mark.parametrize("frozen", [["gripper"], ["wrist"], "wrist_roll", ["wrist_roll", "wrist_roll"]])
def test_bad_frozen_joints_are_refused(tmp_path, frozen):
    cfg = copy.deepcopy(CFG)
    cfg["arm"]["frozen_joints"] = frozen
    with pytest.raises(SystemExit, match="frozen_joints"):
        record.main(["--name", "x", "--config", str(write_cfg(tmp_path, cfg))])


# --- default mode: print only, never run ----------------------------------------------

def test_default_prints_and_never_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted without --run"))
    monkeypatch.setattr(record, "record_live", lambda *a, **k: pytest.fail("recorded without --run"))
    assert record.main(["--name", "demo1", "--episodes", "3", "--config", str(write_cfg(tmp_path))]) == 0
    out, err = capsys.readouterr()
    assert "--dataset.num_episodes=3" in out and "local/demo1" in out
    assert "--run" in err and "frozen" not in err


def test_printed_command_round_trips_through_the_shell(tmp_path, capsys):
    record.main(["--name", "demo1", "--scene", "--config", str(write_cfg(tmp_path)), "--print-only"])
    cmd = shlex.split(capsys.readouterr().out)
    cfg = parse_with_lerobot(cmd)
    assert set(cfg.robot.cameras) == {"wrist", "scene"}
    assert cfg.dataset.single_task == record.TASK


def test_print_only_warns_that_lerobot_record_would_call_a_frozen_servo(tmp_path, capsys):
    assert record.main(["--name", "demo1", "--config", str(write_cfg(tmp_path, FROZEN_CFG)), "--print-only"]) == 0
    out, err = capsys.readouterr()
    assert "local/demo1" in out and "warning" not in out           # stdout stays a clean command
    assert "warning" in err and "wrist_roll" in err and "lerobot-record" in err and "--run" in err


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


def test_find_lerobot_record_falls_back_to_path(monkeypatch):
    seen = []

    def fake_which(name, path=None):
        seen.append(path)
        return None if path else "/on/path/lerobot-record"
    monkeypatch.setattr(record.shutil, "which", fake_which)
    assert record.find_lerobot_record() == "/on/path/lerobot-record"
    assert seen[0] is not None and seen[1] is None


# --- fakes for the native recorder ------------------------------------------------------

READY = {**DEFAULT_POSES["ready"], "gripper": DEFAULT_POSES["gripper_open"]}
ROLL_JUNK = 7.25            # what a fake frozen servo "reads": the recorder must never store it
W, H = 64, 36               # small webcam frames keep the video encoding fast


class VirtualClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def wavy_leader(t):
    """Where the person holds the leader at time t: every joint moving, wrist_roll included."""
    return {**READY, "shoulder_pan": 20 * math.sin(t), "elbow_flex": READY["elbow_flex"] + 15 * math.sin(2 * t),
            "wrist_roll": 40 * math.sin(3 * t), "gripper": 30 + 20 * math.sin(t)}


class Hardware:
    """Shared state of the fake leader, follower, webcams and keyboard, and what they saw."""

    def __init__(self):
        self.clock = VirtualClock()
        self.leader_pose = wavy_leader
        self.events = []                # "align", "command", ... in order
        self.opened = []
        self.closed = []
        self.keys = {}                  # {key-poll number from demo 1 on: line typed}
        self.polls = 0
        self.before_start = []          # key polls before demo 1 (teleop, not recording), then Enter
        self.started = False
        self.all_polls = 0
        self.on_leader_read = None      # called with the read count: raise or press Ctrl-C there
        self.leader_reads = 0
        self.follower = self.leader = None
        self.fail_open = set()
        self.camera_error_at = None     # camera read number that raises (a stalled webcam)
        self.camera_reads = 0
        self.on_command = None          # called with the command number before it completes

    def read_key(self, timeout_s):
        self.all_polls += 1
        if self.all_polls > 100_000:     # a loop that never ends must fail the test, not hang it
            raise AssertionError("the recorder kept polling the keyboard: runaway loop")
        if not self.started:
            if self.before_start:
                return self.before_start.pop(0)
            self.started = True
            return ""                   # Enter: start demo 1
        n = self.polls
        self.polls += 1
        return self.keys.get(n)


class FakeLeader:
    def __init__(self, hw):
        self.hw = hw

    def read(self):
        self.hw.leader_reads += 1
        if self.hw.on_leader_read is not None:
            self.hw.on_leader_read(self.hw.leader_reads)
        return dict(self.hw.leader_pose(self.hw.clock.now()))

    def close(self):
        self.hw.closed.append("leader")


class FakeFollower:
    """Stands in for RealArm: goes exactly where it is commanded and logs every command. A frozen
    joint reads as ROLL_JUNK, so only the recorder can make it a constant."""

    def __init__(self, hw, frozen=()):
        self.hw = hw
        self.frozen = tuple(frozen)
        self.clock = hw.clock
        self.pos = dict(READY)
        self.commands = []

    def read(self):
        return {j: ROLL_JUNK if j in self.frozen else v for j, v in self.pos.items()}

    def command(self, targets):
        self.hw.events.append("command")
        if self.hw.on_command is not None:          # e.g. a Ctrl-C arriving mid-command
            self.hw.on_command(len(self.commands) + 1)
        self.commands.append(dict(targets))         # only a command that wasn't cut short counts
        self.pos.update({j: float(v) for j, v in targets.items()})

    def close(self):
        self.hw.closed.append("follower")


class FakeWebcam:
    def __init__(self, hw, index, width, height, fps):
        self.hw, self.size = hw, (width, height)
        hw.opened.append(("camera", index, width, height, fps))

    def read(self):
        self.hw.camera_reads += 1
        if self.hw.camera_reads == self.hw.camera_error_at:
            raise TimeoutError("latest frame is too old")
        w, h = self.size
        return np.full((h, w, 3), self.hw.camera_reads % 256, dtype=np.uint8)

    def close(self):
        self.hw.closed.append("camera")


class FakeDataset:
    """The LeRobotDataset calls the Recorder makes, in memory. on_save, if set, runs half-way
    through each save (after it started, before it completed)."""

    def __init__(self, features, root="/nonexistent-fake-dataset"):
        self.features = features
        self.root = root
        self.buffer, self.episodes, self.saves = [], [], []
        self.on_save = None

    def add_frame(self, frame):
        self.buffer.append(frame)

    def save_episode(self, parallel_encoding=True):
        self.saves.append(parallel_encoding)
        if self.on_save is not None:
            self.on_save()
        self.episodes.append(self.buffer)
        self.buffer = []

    def clear_episode_buffer(self, delete_images=True):
        self.buffer = []

    def has_pending_frames(self):
        return bool(self.buffer)

    @property
    def num_episodes(self):
        return len(self.episodes)


@pytest.fixture
def no_hardware(monkeypatch):
    """Opening a serial port or a webcam fails the test."""
    import cv2
    import serial

    def refuse(*args, **kwargs):
        raise AssertionError("tried to open hardware")

    monkeypatch.setattr(serial, "Serial", refuse)
    monkeypatch.setattr(cv2, "VideoCapture", refuse)


@pytest.fixture(scope="module")
def features():
    rec = parse_with_lerobot(build(width=W, height=H))
    return record.dataset_features(rec)[1]


def make_recorder(hw, dataset, frozen=(), episodes=2, episode_time_s=1, reset_time_s=1, max_step=1.5, **kw):
    hw.follower = FakeFollower(hw, frozen)
    hw.leader = FakeLeader(hw)
    cams = {"wrist": FakeWebcam(hw, 1, W, H, 30)}
    return record.Recorder(hw.leader, hw.follower, cams, dataset, episodes=episodes, fps=30, max_step=max_step,
                           episode_time_s=episode_time_s, reset_time_s=reset_time_s, frozen=frozen,
                           keys=hw.read_key, log=lambda m: None, **kw)


def column(frames, key, joint):
    return [float(f[key][JOINTS.index(joint)]) for f in frames]


# --- the loop: follows the leader at the speed cap, records lerobot-record's frames --------

def test_frames_hold_all_six_joints_and_the_images(features):
    hw = Hardware()
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=1, reset_time_s=0).run() == 1
    [demo] = ds.episodes
    assert len(demo) == 30                                          # 1 s at 30 fps
    first = demo[0]
    assert set(first) == {"observation.state", "action", "observation.images.wrist", "task"}
    assert first["task"] == record.TASK
    assert first["observation.state"].dtype == np.float32 and first["observation.state"].shape == (6,)
    assert first["observation.images.wrist"].shape == (H, W, 3)
    assert first["observation.state"].tolist() == pytest.approx([READY[j] for j in JOINTS])   # observed first
    assert first["action"].tolist() == pytest.approx([wavy_leader(0.0)[j] for j in JOINTS])   # the leader


def test_follower_steps_towards_the_leader_at_the_speed_cap(features):
    hw = Hardware()
    far = {**READY, "shoulder_pan": 60.0, "elbow_flex": READY["elbow_flex"] - 50, "gripper": 5.0}
    hw.leader_pose = lambda t: far
    ds = FakeDataset(features)
    make_recorder(hw, ds, episodes=1, reset_time_s=0, max_step=0.75).run()
    sent = hw.follower.commands
    prev = READY
    for cmd in sent:
        for j in ("shoulder_pan", "elbow_flex"):
            assert abs(cmd[j] - prev[j]) <= 0.75 + 1e-9
        assert cmd["gripper"] == 5.0                                # the claw goes straight there
        prev = cmd
    assert sent[-1]["shoulder_pan"] == pytest.approx(30 * 0.75)    # 30 capped steps, still on its way
    assert all(f["action"][0] == pytest.approx(60.0) for f in ds.episodes[0])   # the leader is recorded


def test_follow_step_is_teachs_speed_cap():
    from arm import teach

    cmd = dict(READY)
    out = teach.follow_step(cmd, {**READY, "shoulder_pan": 10.0, "shoulder_lift": -200.0, "gripper": 90.0}, 2.0)
    assert out is cmd
    assert (cmd["shoulder_pan"], cmd["shoulder_lift"], cmd["gripper"]) == (2.0, READY["shoulder_lift"] - 2.0, 90.0)


def test_a_frozen_joint_is_never_commanded_and_recorded_as_a_constant(features):
    hw = Hardware()
    ds = FakeDataset(features)
    make_recorder(hw, ds, frozen=("wrist_roll",), episodes=2).run()
    assert hw.follower.commands and all("wrist_roll" not in c for c in hw.follower.commands)
    assert all(set(c) == set(JOINTS) - {"wrist_roll"} for c in hw.follower.commands)
    frames = [f for demo in ds.episodes for f in demo]
    assert len(frames) == 60
    assert set(column(frames, "observation.state", "wrist_roll")) == {FROZEN_READING}   # not ROLL_JUNK
    assert set(column(frames, "action", "wrist_roll")) == {FROZEN_READING}             # not the leader's
    assert len(set(column(frames, "action", "shoulder_pan"))) > 30                      # the others move


def test_without_frozen_joints_wrist_roll_follows_the_leader(features):
    hw = Hardware()
    ds = FakeDataset(features)
    make_recorder(hw, ds, episodes=1, reset_time_s=0).run()
    assert all("wrist_roll" in c for c in hw.follower.commands)
    assert len(set(column(ds.episodes[0], "action", "wrist_roll"))) > 20


def test_the_reset_keeps_following_but_records_nothing(features):
    hw = Hardware()
    ds = FakeDataset(features)
    make_recorder(hw, ds, episodes=2, episode_time_s=1, reset_time_s=2).run()
    assert [len(d) for d in ds.episodes] == [30, 30]
    assert len(hw.follower.commands) == 2 * (30 + 60)              # every reset frame commanded too
    assert ds.saves == [False, False]                              # one camera after another


def test_the_loop_runs_at_fps(features):
    hw = Hardware()
    make_recorder(hw, FakeDataset(features), episodes=1, episode_time_s=2, reset_time_s=0).run()
    assert hw.clock.now() == pytest.approx(2.0)


def test_a_demo_recorded_too_slowly_gets_a_warning(features):
    hw = Hardware()
    hw.on_leader_read = lambda n: hw.clock.sleep(0.05)      # each frame takes 50 ms: 20 fps, not 30
    logs = []
    rec = make_recorder(hw, FakeDataset(features), episodes=1, reset_time_s=0)
    rec.log = logs.append
    rec.run()
    assert any("recorded at 20.0 fps, not 30" in m for m in logs)


def test_before_demo_1_it_follows_without_recording_until_enter(features):
    """The line-up leaves the arms wherever the follower was; the person brings them to `ready`."""
    hw = Hardware()
    hw.before_start = [None] * 20 + ["r"]            # r: nothing to discard yet, still waiting
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=1, reset_time_s=0).run() == 1
    assert len(hw.follower.commands) == 20 + 30
    assert [len(d) for d in ds.episodes] == [30]
    assert ds.episodes[0][0]["observation.state"].tolist() == pytest.approx(
        [hw.follower.commands[19][j] for j in JOINTS])         # recording starts where it got to


def test_q_before_demo_1_records_nothing(features):
    hw = Hardware()
    hw.before_start = [None] * 5 + ["q"]
    ds = FakeDataset(features)
    assert make_recorder(hw, ds).run() == 0
    assert ds.episodes == [] and ds.buffer == [] and len(hw.follower.commands) == 5


# --- keys: Enter, r, q -------------------------------------------------------------------

def test_enter_ends_a_demo_early(features):
    hw = Hardware()
    hw.keys = {10: ""}                      # poll 10: Enter, during demo 1
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=2).run() == 2
    assert [len(d) for d in ds.episodes] == [10, 30]


def test_enter_in_the_reset_starts_the_next_demo(features):
    hw = Hardware()
    hw.keys = {35: ""}                      # demo 1: polls 0-29; reset: 30-59
    ds = FakeDataset(features)
    make_recorder(hw, ds, episodes=2).run()
    assert [len(d) for d in ds.episodes] == [30, 30]
    assert len(hw.follower.commands) == 30 + 5 + 30 + 30


def test_r_discards_the_demo_in_progress_and_records_it_again(features):
    hw = Hardware()
    hw.keys = {12: "r"}
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=2).run() == 2
    assert [len(d) for d in ds.episodes] == [30, 30]
    first = ds.episodes[0][0]["action"][0]                  # demo 1 again, recorded after the reset
    assert first == pytest.approx(wavy_leader((12 + 30) / 30)["shoulder_pan"], abs=1e-4)


def test_r_in_the_reset_discards_the_demo_just_finished(features):
    hw = Hardware()
    hw.keys = {40: "R"}                     # reset after demo 1; case doesn't matter
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=1).run() == 1
    [demo] = ds.episodes
    # demo 1 (polls 0-29) dropped at poll 40; the reset restarts (41-70); demo 1 again from poll 71
    assert demo[0]["action"][0] == pytest.approx(wavy_leader(70 / 30)["shoulder_pan"], abs=1e-4)


def test_q_in_the_reset_saves_the_finished_demo_and_stops(features):
    hw = Hardware()
    hw.keys = {45: "q"}
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=5).run() == 1
    assert [len(d) for d in ds.episodes] == [30]
    assert len(hw.follower.commands) == 45


def test_q_during_a_demo_keeps_the_saved_ones_and_drops_that_demo(features):
    hw = Hardware()
    hw.keys = {70: "q"}                     # demo 2 runs from poll 60
    ds = FakeDataset(features)
    assert make_recorder(hw, ds, episodes=5).run() == 1
    assert [len(d) for d in ds.episodes] == [30] and ds.buffer == []


def test_other_keys_do_nothing(features):
    hw = Hardware()
    hw.keys = {5: "x", 6: "stop"}
    logs = []
    ds = FakeDataset(features)
    rec = make_recorder(hw, ds, episodes=1, reset_time_s=0)
    rec.log = logs.append
    rec.run()
    assert [len(d) for d in ds.episodes] == [30]
    assert any("'x' does nothing" in m for m in logs)


def test_the_last_demo_can_still_be_dropped_in_its_reset(features):
    hw = Hardware()
    hw.keys = {35: "r"}
    ds = FakeDataset(features)
    make_recorder(hw, ds, episodes=1).run()
    assert len(ds.episodes) == 1 and hw.polls > 60                  # recorded again, then saved


def test_read_key_reads_lines_without_blocking(monkeypatch):
    r, w = os.pipe()
    stdin = os.fdopen(r)
    monkeypatch.setattr(record.sys, "stdin", stdin)
    try:
        assert record.read_key(0) is None                           # nothing typed: no wait
        os.write(w, b"R \n")
        assert record.read_key(0.5) == "r"
        os.write(w, b"\n")
        assert record.read_key(0.5) == ""                            # a bare Enter
        os.close(w)
        with pytest.raises(record.RecordError, match="input closed"):
            record.read_key(0.5)
    finally:
        stdin.close()


def test_read_key_from_an_unwatchable_input_is_an_error(monkeypatch):
    class Pseudo:
        def fileno(self):
            import io
            raise io.UnsupportedOperation("redirected stdin is pseudofile, has no fileno()")

    monkeypatch.setattr(record.sys, "stdin", Pseudo())
    with pytest.raises(record.RecordError, match="terminal"):
        record.read_key(0)


# --- Ctrl-C --------------------------------------------------------------------------------

@contextlib.contextmanager
def no_escaping_ctrl_c():
    """A KeyboardInterrupt in here fails the test (pytest would otherwise stop the whole session)."""
    try:
        yield
    except KeyboardInterrupt:
        pytest.fail("a Ctrl-C raised where it must only be counted")


def test_ctrl_c_flag_stops_before_the_next_command(features):
    hw = Hardware()
    stop = record.Stop()
    hw.on_leader_read = lambda n: stop.press() if n == 40 else None    # demo 2's 10th frame
    ds = FakeDataset(features)
    rec = make_recorder(hw, ds, episodes=3, reset_time_s=0, stop=stop)
    with pytest.raises(KeyboardInterrupt):
        rec.run()
    assert len(hw.follower.commands) == 39                  # the frame in which it came sent nothing
    assert [len(d) for d in ds.episodes] == [30] and len(ds.buffer) == 9


def test_ctrl_c_in_the_reset_does_not_save_the_waiting_demo(features):
    hw = Hardware()
    stop = record.Stop()
    hw.on_leader_read = lambda n: stop.press() if n == 35 else None
    ds = FakeDataset(features)
    with pytest.raises(KeyboardInterrupt):
        make_recorder(hw, ds, episodes=2, stop=stop).run()
    assert ds.episodes == [] and len(ds.buffer) == 30       # left for the caller to discard


def test_a_second_ctrl_c_interrupts_at_once():
    stop = record.Stop()
    stop.press()
    assert stop.is_set()
    with pytest.raises(KeyboardInterrupt):
        stop.press()


def test_ctrl_c_handler_is_installed_and_restored():
    before = signal.getsignal(signal.SIGINT)
    stop = record.Stop()
    restore = record.catch_ctrl_c(stop)
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)                        # what a Ctrl-C does, without sending one
        assert stop.is_set()
    finally:
        restore()
    assert signal.getsignal(signal.SIGINT) is before


def test_ctrl_c_presses_only_count_inside_a_protected_section():
    stop = record.Stop()
    with no_escaping_ctrl_c(), stop.protected():
        stop.press()
        stop.press()                                        # would raise outside
        stop.press()
    assert stop.presses == 3 and stop.is_set()
    with pytest.raises(KeyboardInterrupt):
        stop.press()                                        # outside again: a second press interrupts


def test_a_double_ctrl_c_during_a_save_lets_the_save_finish(features):
    hw = Hardware()
    stop = record.Stop()
    ds = FakeDataset(features)
    ds.on_save = lambda: (stop.press(), stop.press())      # both presses arrive while saving demo 1
    logs = []
    rec = make_recorder(hw, ds, episodes=3, reset_time_s=0, stop=stop)
    rec.log = logs.append
    with pytest.raises(KeyboardInterrupt):
        rec.run()
    assert [len(d) for d in ds.episodes] == [30]            # saved whole, then stopped
    assert len(hw.follower.commands) == 30                  # no command after the save
    assert any("saving demo 1" in m and "Ctrl-C" in m for m in logs)
    assert sum("Ctrl-C: stopping" in m for m in logs) == 1  # acknowledged once, by the loop


def test_a_failed_save_marks_the_dataset_broken(features, tmp_path):
    hw = Hardware()
    ds = FakeDataset(features, root=tmp_path)

    def disk_full():
        raise OSError(28, "No space left on device")

    ds.on_save = disk_full
    with pytest.raises(OSError):
        make_recorder(hw, ds, episodes=2, reset_time_s=0).run()
    marker = tmp_path / record.BROKEN_MARKER
    assert "saving demo 1 failed" in marker.read_text() and "No space left" in marker.read_text()


# --- --run end to end, with fake hardware and a real LeRobotDataset --------------------------

@pytest.fixture
def live(tmp_path, monkeypatch, no_hardware):
    """--run with every device faked: datasets under tmp, an interactive terminal whose person
    says the area is clear, and the fake leader, follower, webcams, line-up and keyboard."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(record, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(record, "interactive", lambda: True)
    hw = Hardware()
    hw.prompts = []
    hw.answer = ""
    hw.align = None

    def fake_input(prompt=""):
        hw.prompts.append(prompt)
        return hw.answer

    def leader(port, robot_id):
        if "leader" in hw.fail_open:
            raise RuntimeError("no matching calibration for the leader")
        hw.opened.append(("leader", port, robot_id))
        hw.leader = FakeLeader(hw)
        hw.started = False              # each run starts demo 1 with an Enter
        hw.polls = 0                    # and counts its key polls from demo 1 again
        return hw.leader

    def follower(port, robot_id, max_relative_target, frozen=()):
        if "follower" in hw.fail_open:
            raise RuntimeError("follower not answering")
        hw.opened.append(("follower", port, robot_id, max_relative_target, tuple(frozen)))
        hw.follower = FakeFollower(hw, frozen)
        return hw.follower

    def line_up(lead, foll):
        hw.events.append("align")
        assert foll.commands == []                          # nothing commanded before the line-up
        if hw.align is not None:
            hw.align()

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(real_mod, "RealLeader", leader)
    monkeypatch.setattr(real_mod, "RealArm", follower)
    monkeypatch.setattr(policy_mod, "OpenCVFrames", lambda *a: FakeWebcam(hw, *a))
    monkeypatch.setattr(align_mod, "live_align", line_up)
    monkeypatch.setattr(record, "read_key", hw.read_key)
    hw.config = str(write_cfg(tmp_path, FROZEN_CFG))
    return hw


def run_live(hw, *extra, name="demo1", episodes=2, reset=1):
    return record.main(["--name", name, "--episodes", str(episodes), "--config", hw.config, "--run",
                        "--width", str(W), "--height", str(H), "--episode-time", "1", "--reset-time", str(reset),
                        *extra])


def saved_frames(root: Path):
    import pandas as pd

    info = json.loads((root / "meta" / "info.json").read_text())
    data = pd.read_parquet(root / "data") if (root / "data").exists() else None
    return info, data


def test_run_lines_up_before_the_first_command_and_records(live, capsys):
    assert run_live(live) == 0
    assert live.events[0] == "align" and live.events.count("align") == 1
    assert live.events[1:] and set(live.events[1:]) == {"command"}
    assert len(live.prompts) == 1 and "clear" in live.prompts[0].lower()
    assert [o[0] for o in live.opened] == ["leader", "camera", "follower"]     # read-only ones first
    assert live.opened[2] == ("follower", "/dev/tty.usbmodemF", "so101_follower_arm", 10, ("wrist_roll",))
    assert live.opened[1] == ("camera", 1, W, H, 30)                           # as the command describes
    assert sorted(live.closed) == ["camera", "follower", "leader"]
    info, data = saved_frames(record.DATASETS / "demo1")
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (2, 60, 30)
    roll = JOINTS.index("wrist_roll")
    assert {s[roll] for s in data["observation.state"]} == {FROZEN_READING}
    assert {a[roll] for a in data["action"]} == {FROZEN_READING}
    out = capsys.readouterr().out
    assert "2 demo(s) saved" in out and "holding its pose" in out


def test_run_writes_the_features_lerobot_record_writes(live, tmp_path):
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import DEFAULT_FEATURES
    from lerobot.utils.feature_utils import combine_feature_dicts

    assert run_live(live, "--scene", episodes=1) == 0
    # What lerobot_record.py record() computes for the same command (it never connects to compute them).
    cfg = parse_with_lerobot(build(name="demo1", cameras=["wrist", "scene"], width=W, height=H))
    cfg.robot.calibration_dir = tmp_path / "calibration"
    robot = make_robot_from_config(cfg.robot)
    teleop_proc, _, obs_proc = make_default_processors()
    expected = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_proc, use_videos=cfg.dataset.video,
                                            initial_features=create_initial_features(action=robot.action_features)),
        aggregate_pipeline_dataset_features(pipeline=obs_proc, use_videos=cfg.dataset.video,
                                            initial_features=create_initial_features(
                                                observation=robot.observation_features)))
    info, _ = saved_frames(record.DATASETS / "demo1")
    expected = json.loads(json.dumps(expected))
    written = {k: v for k, v in info["features"].items() if k not in DEFAULT_FEATURES}
    # Saving adds the video encoding details to each camera's "info"; everything else is identical.
    strip = lambda feats: {k: {kk: vv for kk, vv in v.items() if kk != "info"} for k, v in feats.items()}  # noqa: E731
    got = strip(written)
    assert got == strip(expected)
    for key, feature in expected.items():
        assert feature.get("info", {}).items() <= written[key].get("info", {}).items()
    assert got["observation.state"]["names"] == got["action"]["names"] == [f"{j}.pos" for j in JOINTS]
    assert info["robot_type"] == robot.name
    assert ("camera", 2, W, H, 30) in live.opened


def test_ctrl_c_keeps_the_saved_demos_and_sends_no_more_commands(live, capsys):
    def ctrl_c(n):
        if n == 40:                                         # demo 2 (no reset), 10th frame
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)    # the installed handler, as Ctrl-C would
    before = signal.getsignal(signal.SIGINT)
    live.on_leader_read = ctrl_c
    assert run_live(live, episodes=3, reset=0) == 1
    assert len(live.follower.commands) == 39
    assert "follower" in live.closed                         # closed: RealArm keeps the torque on
    root = record.DATASETS / "demo1"
    info, data = saved_frames(root)
    assert (info["total_episodes"], info["total_frames"]) == (1, 30) and len(data) == 30
    assert not (root / "images").exists()                    # the partial demo's frames are gone
    assert signal.getsignal(signal.SIGINT) is before
    captured = capsys.readouterr()
    assert "stopped: Ctrl-C" in captured.err and "1 demo(s) saved" in captured.out
    assert "unsaved demo was discarded" in captured.out


def test_a_keyboard_interrupt_anywhere_ends_the_same_way(live):
    def interrupt(n):
        if n == 50:
            raise KeyboardInterrupt
    live.on_leader_read = interrupt
    assert run_live(live, episodes=3, reset=0) == 1
    assert len(live.follower.commands) == 49
    info, _ = saved_frames(record.DATASETS / "demo1")
    assert info["total_episodes"] == 1


def test_a_failing_webcam_stops_and_keeps_the_saved_demos(live, capsys):
    live.camera_error_at = 45
    assert run_live(live, episodes=3, reset=0) == 1
    assert len(live.follower.commands) == 44                 # the failed frame sent nothing
    assert "TimeoutError" in capsys.readouterr().err
    info, _ = saved_frames(record.DATASETS / "demo1")
    assert info["total_episodes"] == 1


def test_stopped_before_any_demo_leaves_no_dataset_behind(live, capsys):
    def ctrl_c_while_lining_up():
        raise KeyboardInterrupt
    live.align = ctrl_c_while_lining_up
    assert run_live(live) == 1
    assert live.follower.commands == []
    assert not (record.DATASETS / "demo1").exists()          # so the same --name can run again
    assert "nothing recorded" in capsys.readouterr().out
    live.align = None
    assert run_live(live, episodes=1) == 0


def test_a_failing_line_up_moves_nothing(live, capsys):
    def closed():
        raise align_mod.AlignError("input closed before Enter was pressed; nothing moved")
    live.align = closed
    assert run_live(live) == 1
    assert live.follower.commands == []
    assert "nothing moved" in capsys.readouterr().err


def test_leader_failure_never_opens_the_follower(live, capsys):
    live.fail_open.add("leader")
    assert run_live(live) == 1
    assert live.opened == [] and "calibration for the leader" in capsys.readouterr().err
    assert not (record.DATASETS / "demo1").exists()


def test_follower_failure_closes_the_rest(live):
    live.fail_open.add("follower")
    assert run_live(live) == 1
    assert sorted(live.closed) == ["camera", "leader"]


@pytest.mark.parametrize("answer", ["n", "no", " x "])
def test_run_cancelled_by_any_other_answer(live, answer):
    live.answer = answer
    with pytest.raises(SystemExit, match="cancelled"):
        run_live(live)
    assert live.opened == [] and not record.DATASETS.joinpath("demo1").exists()


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_run_cancelled_by_eof_or_ctrl_c(live, monkeypatch, exc):
    def raise_(*a):
        raise exc
    monkeypatch.setattr("builtins.input", raise_)
    with pytest.raises(SystemExit, match="cancelled"):
        run_live(live)
    assert live.opened == []


def test_run_refused_without_a_terminal(live, monkeypatch):
    monkeypatch.setattr(record, "interactive", lambda: False)
    with pytest.raises(SystemExit, match="interactive terminal"):
        run_live(live)
    assert live.opened == [] and live.prompts == []


def test_run_never_needs_lerobot_record(live, monkeypatch):
    monkeypatch.setattr(record, "find_lerobot_record", lambda: None)
    assert run_live(live, episodes=1) == 0


# --- --resume and the overwrite guard ---------------------------------------------------------

def test_run_refuses_to_overwrite_an_existing_dataset(live):
    (record.DATASETS / "demo1").mkdir(parents=True)
    (record.DATASETS / "demo1" / "mine.txt").write_text("keep")
    with pytest.raises(SystemExit, match="already exists"):
        run_live(live)
    assert live.opened == [] and live.prompts == []
    assert (record.DATASETS / "demo1" / "mine.txt").read_text() == "keep"


def test_resume_of_missing_dataset_is_refused(live):
    with pytest.raises(SystemExit, match="nothing to resume"):
        run_live(live, "--resume")
    assert live.opened == [] and live.prompts == []
    assert not (record.DATASETS / "demo1").exists()


def test_resume_adds_demos_to_the_dataset(live, capsys):
    assert run_live(live, episodes=1) == 0
    assert run_live(live, "--resume", episodes=2) == 0
    info, data = saved_frames(record.DATASETS / "demo1")
    assert (info["total_episodes"], info["total_frames"]) == (3, 90)
    assert sorted(set(data["episode_index"])) == [0, 1, 2]
    assert "has 3" in capsys.readouterr().out


def test_resume_refuses_a_dataset_recorded_differently(live):
    assert run_live(live, episodes=1) == 0
    live.opened.clear()
    with pytest.raises(SystemExit, match="recorded differently"):
        run_live(live, "--resume", "--scene", episodes=1)          # a second camera now
    assert live.opened == []                                       # before anything is opened
    info, _ = saved_frames(record.DATASETS / "demo1")
    assert info["total_episodes"] == 1


# --- review fixes: the speed cap end to end, Ctrl-C during writes, untouched resumes ---------------

def snapshot(root: Path) -> dict:
    """Every file's bytes and every folder under root, to prove a run left them as they were."""
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None) for p in sorted(root.rglob("*"))}


def press_ctrl_c():
    """What Ctrl-C does to this process, without sending a real SIGINT: call the installed handler."""
    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)


def run_live_catching_ctrl_c(hw, *extra, **kw):
    """run_live, where a KeyboardInterrupt escaping it (a Ctrl-C that cut a write short) fails the
    test instead of stopping the whole pytest session."""
    with no_escaping_ctrl_c():
        return run_live(hw, *extra, **kw)


def test_run_steps_the_follower_at_speed_over_fps(live):
    """main must hand record_live the per-frame step (speed / fps), not the speed itself."""
    live.leader_pose = lambda t: {**READY, "shoulder_pan": 60.0}
    assert run_live(live, "--speed", "25", episodes=1, reset=0) == 0
    pans = [READY["shoulder_pan"]] + [c["shoulder_pan"] for c in live.follower.commands]
    steps = [b - a for a, b in zip(pans, pans[1:])]
    assert steps and all(s == pytest.approx(25 / 30) for s in steps)


def test_run_routes_ctrl_c_to_a_clean_stop_between_commands(live, capsys):
    """record_live installs the Stop handler: a Ctrl-C that lands in the middle of a command lets
    that command finish (Python's default handler would raise inside it), and the loop stops
    before the next one."""
    seen = []

    def ctrl_c_mid_command(n):
        if n == 40:                                         # demo 2, 10th frame (no reset)
            seen.append(signal.getsignal(signal.SIGINT))
            press_ctrl_c()

    live.on_command = ctrl_c_mid_command
    assert run_live_catching_ctrl_c(live, episodes=3, reset=0) == 1
    assert seen[0] is not signal.default_int_handler and isinstance(getattr(seen[0], "__self__", None), record.Stop)
    assert len(live.follower.commands) == 40                # command 40 finished, nothing after it
    assert live.events.count("command") == 40
    info, _ = saved_frames(record.DATASETS / "demo1")
    assert info["total_episodes"] == 1
    out = capsys.readouterr().out
    assert out.count("Ctrl-C: stopping") == 1


def lerobot_writer():
    from lerobot.datasets.dataset_writer import DatasetWriter

    return DatasetWriter


def assert_consistent(root: Path, episodes: int, frames: int):
    """meta/info.json, the parquet rows and the leftover frames all agree."""
    info, data = saved_frames(root)
    assert (info["total_episodes"], info["total_frames"]) == (episodes, frames)
    assert len(data) == frames and sorted(set(data["episode_index"])) == list(range(episodes))
    assert not list(root.rglob("*.png"))


def test_a_double_ctrl_c_mid_save_leaves_a_consistent_dataset(live, monkeypatch):
    writer = lerobot_writer()
    real_save = writer._save_episode_data
    calls = []

    def save_then_ctrl_c_twice(self, buffer):
        out = real_save(self, buffer)                       # the parquet rows are written ...
        calls.append(1)
        if len(calls) == 1:
            press_ctrl_c()                                  # ... and Ctrl-C twice before the save ends
            press_ctrl_c()
        return out

    monkeypatch.setattr(writer, "_save_episode_data", save_then_ctrl_c_twice)
    assert run_live_catching_ctrl_c(live, episodes=3, reset=0) == 1
    root = record.DATASETS / "demo1"
    assert_consistent(root, 1, 30)
    assert not (root / record.BROKEN_MARKER).exists()
    assert run_live(live, "--resume", episodes=1, reset=0) == 0       # and it still resumes cleanly
    assert_consistent(root, 2, 60)


def test_a_double_ctrl_c_mid_finalize_still_finalizes(live, monkeypatch):
    from lerobot.datasets import LeRobotDataset

    real_finalize = LeRobotDataset.finalize

    def ctrl_c_twice_then_finalize(self):
        press_ctrl_c()
        press_ctrl_c()
        return real_finalize(self)

    monkeypatch.setattr(LeRobotDataset, "finalize", ctrl_c_twice_then_finalize)
    assert run_live_catching_ctrl_c(live, episodes=2) == 0
    assert_consistent(record.DATASETS / "demo1", 2, 60)


def test_a_failing_save_marks_the_dataset_and_resume_refuses_it(live, monkeypatch, capsys):
    writer = lerobot_writer()
    real_save = writer._save_episode_data
    calls = []

    def second_save_fails(self, buffer):
        calls.append(1)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_save(self, buffer)

    monkeypatch.setattr(writer, "_save_episode_data", second_save_fails)
    assert run_live(live, episodes=3, reset=0) == 1
    root = record.DATASETS / "demo1"
    assert "saving demo 2 failed" in (root / record.BROKEN_MARKER).read_text()
    out = capsys.readouterr().out
    assert "marked broken" in out and "new --name" in out
    assert saved_frames(root)[0]["total_episodes"] == 1
    before = snapshot(root)
    with pytest.raises(SystemExit, match="won't resume.*new --name"):
        run_live(live, "--resume", episodes=1)
    assert snapshot(root) == before


def test_a_create_that_fails_after_its_mkdir_does_not_block_the_name(live, monkeypatch):
    from lerobot.datasets import LeRobotDataset

    real_create = LeRobotDataset.create

    def mkdir_then_fail(*args, root=None, **kwargs):
        Path(root).mkdir(parents=True)
        raise ValueError("bad feature spec")

    monkeypatch.setattr(LeRobotDataset, "create", mkdir_then_fail)
    with pytest.raises(SystemExit) as exc:
        run_live(live)
    assert "bad feature spec" in str(exc.value) and "recorded differently" not in str(exc.value)
    assert not (record.DATASETS / "demo1").exists() and live.opened == []
    monkeypatch.setattr(LeRobotDataset, "create", real_create)
    assert run_live(live, episodes=1) == 0


@pytest.fixture
def resumed(live):
    """A dataset with one demo, and a snapshot of its files, ready for a --resume run."""
    assert run_live(live, episodes=1) == 0
    root = record.DATASETS / "demo1"
    live.opened.clear()
    return root, snapshot(root)


def test_resume_then_q_before_demo_1_leaves_the_dataset_untouched(live, resumed):
    root, before = resumed
    live.before_start = ["q"]
    assert run_live(live, "--resume") == 0
    assert root.exists() and snapshot(root) == before


def test_resume_then_q_mid_demo_leaves_the_dataset_untouched(live, resumed):
    root, before = resumed
    live.keys = {10: "q"}                                   # 10 frames into the new demo
    assert run_live(live, "--resume") == 0
    assert snapshot(root) == before


def test_resume_then_ctrl_c_mid_demo_leaves_the_dataset_untouched(live, resumed):
    root, before = resumed
    live.on_leader_read = lambda n: press_ctrl_c() if n == live.leader_reads_at_start + 10 else None
    live.leader_reads_at_start = live.leader_reads
    assert run_live(live, "--resume") == 1
    assert snapshot(root) == before


def test_resume_then_a_leader_failure_leaves_the_dataset_untouched(live, resumed):
    root, before = resumed
    live.fail_open.add("leader")
    assert run_live(live, "--resume") == 1
    assert snapshot(root) == before


# --- slow: record two fake demos into a real dataset, then train ACT on it -------------------

@pytest.mark.slow
def test_recorded_demos_train_act(live, tmp_path):
    from training import rehearse_sim

    def leader_path(t):
        """ready -> at_object with the claw opening, then closing; the wrist turning all along."""
        start, goal = DEFAULT_POSES["ready"], DEFAULT_POSES["at_object"]
        k = min(1.0, t / 0.6)
        pose = {j: start[j] + (goal[j] - start[j]) * k for j in JOINTS}
        pose["gripper"] = 60.0 if t < 0.7 else 2.0
        pose["wrist_roll"] = 30 * math.sin(4 * t)
        return pose

    live.leader_pose = leader_path
    assert run_live(live, "--width", "320", "--height", "180", episodes=2, reset=0) == 0
    root = record.DATASETS / "demo1"
    info, _ = saved_frames(root)
    assert (info["total_episodes"], info["total_frames"]) == (2, 60)

    steps = 4
    out = tmp_path / "train"
    home = tmp_path / "hf"
    env = dict(os.environ, HF_HOME=str(home), HF_LEROBOT_HOME=str(home / "lerobot"), HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", TORCH_HOME=str(tmp_path / "torch"))
    cmd = [rehearse_sim.find_lerobot_train(), "--dataset.repo_id=local/demo1", f"--dataset.root={root}",
           "--policy.type=act", f"--policy.device={policy_mod.default_device()}", "--policy.push_to_hub=false",
           f"--output_dir={out}", "--job_name=demo1", f"--steps={steps}", f"--save_freq={steps}",
           "--wandb.enable=false", *rehearse_sim.mac_overrides(steps)]
    t0 = time.monotonic()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    log = r.stdout + r.stderr
    print(f"\nlerobot-train {steps} steps took {time.monotonic() - t0:.0f} s")
    assert r.returncode == 0, log[-6000:]
    assert "End of training" in log
    losses = [float(line.split("loss:")[1].split()[0]) for line in log.splitlines() if " loss:" in line]
    assert losses and all(math.isfinite(v) for v in losses), losses     # the constant wrist_roll is no NaN
    print(f"losses: {losses}")
    config = out / "checkpoints" / f"{steps:06d}" / "pretrained_model" / "config.json"
    inputs = json.loads(config.read_text())["input_features"]
    assert inputs["observation.state"]["shape"] == [6]
    assert inputs["observation.images.wrist"]["shape"] == [3, 180, 320]
