"""The simulated ACT pipeline rehearsal (task T2). The fast tests pin the joins between
training/record.py, training/remote.sh and arm/policy.py without training anything; the slow
test runs the whole rehearsal (record, convert, train with lerobot-train, load, run) and is
skipped unless RUN_SLOW=1. No hardware, no network; everything is written under tmp_path."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from arm import pick
from arm import policy as policy_mod
from arm.poses import JOINTS
from arm.sim import DEFAULT_POSES
from training import record, rehearse_sim

REPO = Path(__file__).resolve().parent.parent
REMOTE_SH_USAGE = (REPO / "training" / "remote.sh").read_text().split("set -euo")[0]


@pytest.fixture
def no_hardware(monkeypatch):
    """Opening a serial port or a webcam fails the test."""
    import cv2
    import serial

    def refuse(*args, **kwargs):
        raise AssertionError("tried to open hardware")

    monkeypatch.setattr(serial, "Serial", refuse)
    monkeypatch.setattr(cv2, "VideoCapture", refuse)


# --- record.py's dataset is what arm/policy.py expects ------------------------------------

@pytest.mark.parametrize("cameras", [["wrist"], ["wrist", "scene"]])
def test_record_features_are_lerobot_records_and_line_up_with_the_policy(tmp_path, no_hardware, cameras):
    pytest.importorskip("lerobot.scripts.lerobot_record")
    rec = rehearse_sim.record_config(cameras, record.WIDTH, record.HEIGHT)
    robot, features = rehearse_sim.record_features(rec, tmp_path / "calibration")
    assert not robot.is_connected
    from lerobot.robots.so_follower import SO101FollowerConfig

    assert isinstance(rec.robot, SO101FollowerConfig) and robot.name == "so_follower"
    assert (rec.dataset.fps, rec.dataset.video) == (record.FPS, True)
    assert set(features) == {"action", "observation.state", *(f"observation.images.{c}" for c in cameras)}
    # arm/policy.py builds the state and reads the action in arm/poses.py JOINTS order.
    assert features["observation.state"]["names"] == [f"{j}.pos" for j in JOINTS]
    assert features["action"]["names"] == [f"{j}.pos" for j in JOINTS]
    assert features["observation.state"]["dtype"] == features["action"]["dtype"] == "float32"
    for cam in cameras:
        f = features[f"observation.images.{cam}"]
        assert (f["dtype"], tuple(f["shape"])) == ("video", (record.HEIGHT, record.WIDTH, 3))
        assert f["names"] == ["height", "width", "channels"]
    assert tmp_path.joinpath("calibration").is_dir()        # never ~/.cache's calibration folder


def test_the_policy_runs_at_the_recording_rate():
    """No checkpoint records its fps: pick.py steps the policy at RATE_HZ and opens the webcams
    at RATE_HZ, which must be the rate the demos were recorded at."""
    assert record.FPS == pick.RATE_HZ
    cmd = record.build_command(rehearse_sim.SIM_ROBOT_YAML, "x", 1, ["wrist"], record.WIDTH, record.HEIGHT,
                               record.FPS, 20, 10, False, "lerobot-record")
    assert f"--dataset.fps={pick.RATE_HZ}" in cmd


# --- the simulated demos -----------------------------------------------------------------

def test_render_shows_the_object_moving_with_the_arm():
    ready, at = dict(DEFAULT_POSES["ready"]), dict(DEFAULT_POSES["at_object"])
    img = rehearse_sim.render("wrist", ready, 320, 180)
    assert (img.shape, img.dtype) == ((180, 320, 3), np.uint8)
    red = lambda im: np.argwhere((im == rehearse_sim.OBJECT_RGB).all(-1))
    near, far = red(rehearse_sim.render("wrist", at, 320, 180)), red(img)
    assert len(near) > 4 * len(far) > 0                                   # bigger as the claw comes down
    left = red(rehearse_sim.render("wrist", {**at, "shoulder_pan": 8.0}, 320, 180))
    assert left[:, 1].mean() < near[:, 1].mean() - 20                       # off-centre when panned away
    closed = rehearse_sim.render("wrist", {**at, "gripper": 2.0}, 320, 180)
    assert not np.array_equal(closed, rehearse_sim.render("wrist", at, 320, 180))
    assert rehearse_sim.render("scene", at, 320, 180).shape == (180, 320, 3)


def test_render_is_the_same_picture_at_twice_the_size():
    """So a 640x360 recording converted to 320x180 looks like a 320x180 one."""
    from training import convert

    state = dict(DEFAULT_POSES["at_object"])
    big = convert.resize(rehearse_sim.render("wrist", state, 640, 360), 320, 180)
    small = rehearse_sim.render("wrist", state, 320, 180)
    assert np.abs(big.astype(int) - small.astype(int)).mean() < 3


class Collect:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


def test_a_demo_is_recorded_like_lerobot_record_and_ends_on_the_object(tmp_path, no_hardware):
    pytest.importorskip("lerobot.scripts.lerobot_record")
    rec = rehearse_sim.record_config(["wrist", "scene"], 32, 18)
    _, features = rehearse_sim.record_features(rec, tmp_path)
    ds = Collect()
    n = rehearse_sim.record_demo(ds, features, {"wrist": (32, 18), "scene": (32, 18)}, record.TASK, -8.0, 30)
    assert n == len(ds.frames) == len(rehearse_sim.leader_path(-8.0, 30))
    first, last = ds.frames[0], ds.frames[-1]
    assert set(first) == set(features) | {"task"} and first["task"] == record.TASK
    assert first["observation.state"].tolist() == [DEFAULT_POSES["ready"][j] for j in JOINTS]
    assert first["observation.images.wrist"].shape == (18, 32, 3)
    assert last["action"][-1] == DEFAULT_POSES["gripper_closed"]             # the leader's claw: shut
    assert last["observation.state"][-1] > DEFAULT_POSES["gripper_closed"] + pick.GRIP_MARGIN   # on the object
    assert last["action"][0] == pytest.approx(-8.0)                          # panned to the object


def test_leader_moves_at_demo_speed():
    path = rehearse_sim.leader_path(8.0, 30)
    steps = [max(abs(b[j] - a[j]) for j in JOINTS[:5]) for a, b in zip(path, path[1:])]
    assert max(steps) <= rehearse_sim.DEMO_SPEED_DEG_S / 30 + 1e-9
    assert path[-1]["gripper"] == DEFAULT_POSES["gripper_closed"]


# --- training through remote.sh ------------------------------------------------------------

FAKE_REAL_TRAIN = """#!/usr/bin/env bash
printf '%s\\n' "$@"
"""


def test_shim_passes_remote_sh_flags_with_the_mac_changes(tmp_path):
    real = tmp_path / "real-lerobot-train"
    real.write_text(FAKE_REAL_TRAIN)
    real.chmod(0o755)
    shim = tmp_path / "venv" / "bin" / "lerobot-train"
    rehearse_sim.write_shim(shim, str(real), "mps", 20)
    tfs = '{"brightness": {"kwargs": {"brightness": [0.8, 1.2]}}}'
    args = ["--policy.type=act", "--policy.device=cuda", f"--dataset.image_transforms.tfs={tfs}", "--steps=20"]
    r = subprocess.run([str(shim), *args], env=dict(os.environ, HOME=str(tmp_path)), capture_output=True,
                       text=True, check=True)
    assert r.stdout.splitlines() == ["--policy.type=act", "--policy.device=mps",
                                     f"--dataset.image_transforms.tfs={tfs}", "--steps=20",
                                     *rehearse_sim.mac_overrides(20)]
    assert (tmp_path / "lerobot-train.args").read_text().splitlines() == args



def test_policy_docs_point_where_remote_sh_pulls_checkpoints():
    """remote.sh pull NAME copies each checkpoint to data/checkpoints/NAME/STEP/pretrained_model
    (test_remote.py checks it); arm/policy.py's example used to leave out NAME."""
    import re

    assert re.search(r'load_policy\("data/checkpoints/[^/"]+/\d{6}/pretrained_model"\)', policy_mod.__doc__)
    assert "pull NAME" in REMOTE_SH_USAGE and "data/checkpoints/NAME/" in REMOTE_SH_USAGE

def test_pc_env_keeps_everything_under_out_and_offline(tmp_path):
    env = rehearse_sim.pc_env(tmp_path)
    assert env["FP_TRAIN_HOST"] == "local"
    for key in ("HOME", "HF_HOME", "HF_LEROBOT_HOME", "FP_CHECKPOINT_DIR", "TORCH_HOME"):
        assert Path(env[key]).is_relative_to(tmp_path), key
    assert (env["HF_HUB_OFFLINE"], env["HF_DATASETS_OFFLINE"], env["WANDB_MODE"]) == ("1", "1", "disabled")


# --- loading: the webcams pick.py opens are the simulated ones ----------------------------

class StubAct:
    def __init__(self, path, cameras, device=None):
        self.cameras = cameras


def test_sim_hardware_stands_in_for_the_webcams_and_robot_yaml(tmp_path, monkeypatch, no_hardware):
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"type": "act", "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 180, 320]},
        "observation.images.scene": {"type": "VISUAL", "shape": [3, 180, 320]}}}))
    monkeypatch.setattr(policy_mod, "ActGrasp", StubAct)
    real_frames, real_config = policy_mod.OpenCVFrames, pick.DEFAULT_CONFIG_PATH
    holder = {"arm": rehearse_sim.SimArm()}
    with rehearse_sim.sim_hardware(tmp_path, holder) as webcam:
        policy = pick.load_policy(ckpt, capture_size=(640, 360))
        assert {n: (c.name, c.size, c.fps) for n, c in policy.cameras.items()} == {
            "wrist": ("wrist", (640, 360), pick.RATE_HZ), "scene": ("scene", (640, 360), pick.RATE_HZ)}
        frame = policy.cameras["wrist"].read()
        assert frame.shape == (360, 640, 3)
        assert np.array_equal(frame, rehearse_sim.render("wrist", holder["arm"].read(), 640, 360))
        assert len(webcam.opened) == 2
    assert (policy_mod.OpenCVFrames, pick.DEFAULT_CONFIG_PATH) == (real_frames, real_config)


class StubPolicy:
    def __init__(self, sizes, state=(6,), action=(6,)):
        from types import SimpleNamespace

        self.image_sizes = sizes
        self.config = SimpleNamespace(input_features={"observation.state": SimpleNamespace(shape=state)},
                                      output_features={"action": SimpleNamespace(shape=action)})


def write_info(root, cams, names=None):
    names = names or [f"{j}.pos" for j in JOINTS]
    features = {"observation.state": {"names": names}, "action": {"names": names}}
    features.update({f"observation.images.{c}": {"shape": [h, w, 3]} for c, (w, h) in cams.items()})
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    return root


@pytest.mark.parametrize("policy, cams, names, words", [
    (StubPolicy({"wrist": (640, 360)}), {"wrist": (320, 180)}, None, "cameras"),
    (StubPolicy({"wrist": (320, 180)}), {"wrist": (320, 180), "scene": (320, 180)}, None, "cameras"),
    (StubPolicy({"wrist": (320, 180)}), {"wrist": (320, 180)}, ["gripper.pos", *[f"{j}.pos" for j in JOINTS[:5]]],
     "JOINTS"),
    (StubPolicy({"wrist": (320, 180)}, action=(7,)), {"wrist": (320, 180)}, None, "action"),
])
def test_checkpoint_check_catches_a_mismatch(tmp_path, policy, cams, names, words):
    root = write_info(tmp_path / "ds", cams, names)
    with pytest.raises(RuntimeError, match=words):
        rehearse_sim.check_checkpoint(policy, root)


def test_checkpoint_check_passes_when_everything_lines_up(tmp_path):
    root = write_info(tmp_path / "ds", {"wrist": (320, 180)})
    assert rehearse_sim.check_checkpoint(StubPolicy({"wrist": (320, 180)}), root) == "wrist 320x180"


def test_out_must_be_new_or_empty(tmp_path):
    (tmp_path / "x").write_text("mine")
    with pytest.raises(SystemExit, match="not empty"):
        rehearse_sim.main(["--out", str(tmp_path)])
    assert (tmp_path / "x").read_text() == "mine"


# --- the whole rehearsal ---------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("extra, converts", [([], False), (["--scene", "--record-size", "640x360"], True)],
                         ids=["default", "scene-640x360-convert"])
def test_whole_pipeline_rehearsal(tmp_path, extra, converts):
    """record -> (convert) -> lerobot-train through remote.sh -> pull -> load -> pick_to_hand, in a
    subprocess exactly as `python -m training.rehearse_sim` runs, with every file under tmp_path."""
    out = tmp_path / "rehearsal"
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "training.rehearse_sim", "--steps", "20", "--out", str(out), *extra],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)
    seconds = time.monotonic() - t0
    report = r.stdout[r.stdout.rfind("stage     outcome"):]
    print(f"\n{report}\n(rehearsal took {seconds:.0f} s)")
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-6000:]
    outcomes = {line.split()[0]: line.split()[1] for line in report.splitlines()[1:] if line.strip()}
    assert outcomes == {"record": "ok", "convert": "ok" if converts else "skipped", "train": "ok", "load": "ok",
                        "run": "ok"}
    assert "pick_to_hand -> " in report
    ckpts = list((out / "checkpoints" / rehearse_sim.NAME).glob("*/pretrained_model/model.safetensors"))
    assert len(ckpts) == 1
    trained = json.loads((ckpts[0].parent / "train_config.json").read_text())["policy"]
    assert trained["type"] == "act" and trained["pretrained_backbone_weights"] is None     # nothing downloaded
    config = json.loads((ckpts[0].parent / "config.json").read_text())
    assert (config["device"], config["pretrained_backbone_weights"]) == ("cuda", "ResNet18_Weights.IMAGENET1K_V1")
    image = config["input_features"]["observation.images.wrist"]["shape"]
    assert image == [3, record.HEIGHT, record.WIDTH]
    assert ("observation.images.scene" in config["input_features"]) == ("--scene" in extra)
    assert not (REPO / "data" / "datasets" / rehearse_sim.NAME).exists()
    assert not (REPO / "data" / "checkpoints" / rehearse_sim.NAME).exists()
