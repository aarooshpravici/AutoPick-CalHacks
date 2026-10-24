"""Tests for the ACT grasp mode (task T3). No hardware: cameras are fakes or grey frames,
the arm is simulated, and the checkpoint is a tiny untrained one built here."""

import json
import shutil

import numpy as np
import pytest
import yaml

import arm.policy as policy_mod
import arm.real as real_mod
from arm import pick, poses as poses_mod
from arm.sim import DEFAULT_POSES, SimArm

DT = 1.0 / pick.RATE_HZ

# Same keys as config/robot.yaml (task S6); the real file is never read by the tests.
ROBOT_YAML = {
    "arm": {"follower_port": "/dev/tty.usbmodemTEST1", "leader_port": "/dev/tty.usbmodemTEST2",
            "follower_id": "so101_follower_arm", "leader_id": "so101_leader_arm",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
    "handoff": {"timeout_s": 60, "grip": "squeeze"},
}


def write_config(tmp_path, cameras=None):
    data = json.loads(json.dumps(ROBOT_YAML))
    if cameras is not None:
        data["cameras"] = cameras
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def last_result(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RESULT "))


# --- a real (tiny, untrained) ACT checkpoint -----------------------------------

@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A small ACT checkpoint shaped like ours: 6-joint state/action, wrist camera 320x180."""
    pytest.importorskip("torch")
    pytest.importorskip("lerobot.policies")
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = ACTConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,)),
                        "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 180, 320))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        pretrained_backbone_weights=None, dim_model=32, n_heads=2, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, chunk_size=5, n_action_steps=5, use_vae=False, device="cpu")
    one, zero = torch.ones, torch.zeros
    stats = {"observation.state": {"mean": zero(6), "std": one(6) * 30, "min": -one(6) * 100, "max": one(6) * 100},
             "action": {"mean": zero(6), "std": one(6) * 30, "min": -one(6) * 100, "max": one(6) * 100},
             "observation.images.wrist": {"mean": zero(3, 1, 1) + 0.5, "std": one(3, 1, 1) * 0.2,
                                          "min": zero(3, 1, 1), "max": one(3, 1, 1)}}
    path = tmp_path_factory.mktemp("ckpt") / "pretrained_model"
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    get_policy_class("act")(cfg).save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return path


def test_checkpoint_image_sizes(checkpoint):
    assert policy_mod.checkpoint_image_sizes(checkpoint) == {"wrist": (320, 180)}


def test_act_step_gives_six_joint_targets(checkpoint):
    from arm.policy import ActGrasp, BlankFrames

    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames(640, 360)}, device="cpu")   # resized to 320x180
    policy.reset()
    action = policy.step(dict(DEFAULT_POSES["ready"]))
    assert set(action) == set(poses_mod.JOINTS)
    assert all(np.isfinite(v) for v in action.values())


@pytest.mark.parametrize("size", [(640, 480), (320, 240), (180, 320)])
def test_frames_of_another_aspect_ratio_are_refused_not_squashed(checkpoint, size):
    from arm.policy import ActGrasp, BlankFrames

    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames(*size)}, device="cpu")
    with pytest.raises(ValueError, match="squash"):
        policy.step(dict(DEFAULT_POSES["ready"]))


def test_missing_camera_is_an_error(checkpoint):
    from arm.policy import ActGrasp

    with pytest.raises(ValueError, match="wrist"):
        ActGrasp(checkpoint, cameras={}, device="cpu")


def test_backbone_weights_are_never_downloaded(checkpoint, tmp_path, monkeypatch):
    """Real checkpoints name the ImageNet ResNet weights; loading must not fetch them."""
    import torch
    import torchvision.models._api as tv_api
    from arm.policy import ActGrasp, BlankFrames

    path = tmp_path / "pretrained_model"
    shutil.copytree(checkpoint, path)
    cfg = json.loads((path / "config.json").read_text())
    cfg["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
    (path / "config.json").write_text(json.dumps(cfg))

    def no_download(*args, **kwargs):
        raise AssertionError("tried to load backbone weights from the network or hub cache")

    monkeypatch.setattr(tv_api, "load_state_dict_from_url", no_download)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_download)
    policy = ActGrasp(path, cameras={"wrist": BlankFrames(320, 180)}, device="cpu")
    assert policy.config.pretrained_backbone_weights is None
    assert set(policy.step(dict(DEFAULT_POSES["ready"]))) == set(poses_mod.JOINTS)


def _mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _mps_available(), reason="needs the Mac GPU (mps)")
def test_act_runs_on_mps(checkpoint):
    from arm.policy import ActGrasp, BlankFrames

    assert policy_mod.default_device() == "mps"
    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames(320, 180)})     # default device
    assert policy.device.type == "mps"
    policy.reset()
    action = policy.step(dict(DEFAULT_POSES["ready"]))
    assert set(action) == set(poses_mod.JOINTS)
    assert all(np.isfinite(v) for v in action.values())


def test_untrained_policy_runs_a_whole_pick(checkpoint):
    from arm.policy import ActGrasp, BlankFrames

    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames(320, 180)}, device="cpu")
    r = pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=policy)
    assert r["grasp"] == "policy"
    assert r["result"] in {"in_tray", "missed", "dropped", "stuck"}


def test_cli_dry_run_with_checkpoint(checkpoint, tmp_path, capsys):
    rc = pick.main(["run", "--dry-run", "--policy", str(checkpoint), "--poses", str(tmp_path / "none"),
                    "--config", str(write_config(tmp_path))])
    result = last_result(capsys)
    assert result["grasp"] == "policy"
    assert result["result"] != "error", result["detail"]
    assert rc == pick.EXIT_CODES[result["result"]]


@pytest.mark.parametrize("size, ok", [("640x360", True), ("640x480", False)])
def test_cli_dry_run_camera_size(checkpoint, tmp_path, capsys, size, ok):
    rc = pick.main(["run", "--dry-run", "--policy", str(checkpoint), "--camera-size", size,
                    "--poses", str(tmp_path / "none"), "--config", str(write_config(tmp_path))])
    result = last_result(capsys)
    if ok:
        assert result["result"] != "error", result["detail"]
    else:
        assert (rc, result["result"]) == (1, "error")
        assert "aspect ratio" in result["detail"]


def test_cli_dry_run_checks_robot_yaml_cameras(checkpoint, tmp_path, capsys):
    rc = pick.main(["run", "--dry-run", "--policy", str(checkpoint), "--poses", str(tmp_path / "none"),
                    "--config", str(write_config(tmp_path, cameras={"scene": 2}))])
    result = last_result(capsys)
    assert (rc, result["result"]) == (1, "error")
    assert "wrist" in result["detail"]


# --- the grasp-done rule, the time limit and pacing (stand-in policies) ---------

class ScriptedPolicy:
    """Stands in for ACT: heads to at_object, then asks for a closed claw."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0

    def step(self, state):
        self.steps += 1
        target = dict(DEFAULT_POSES["at_object"])
        near = all(abs(state[j] - target[j]) < 2 for j in poses_mod.ARM_JOINTS)
        target["gripper"] = DEFAULT_POSES["gripper_closed"] if near else DEFAULT_POSES["gripper_open"]
        return target


class StayPolicy:
    """Stays at ready and asks for a fixed claw target."""

    def __init__(self, grip):
        self.grip = grip

    def step(self, state):
        return {**DEFAULT_POSES["ready"], "gripper": self.grip}


def test_policy_grasp_on_object():
    arm = SimArm()
    r = pick.pick_to_tray(arm, DEFAULT_POSES, policy=ScriptedPolicy())
    assert (r["result"], r["grasp"], r["attempts"]) == ("in_tray", "policy", 1)
    assert arm.object == "tray"


def test_policy_closing_on_nothing_is_missed_not_dropped():
    """The claw starts open and travels shut through the 'holding' band: a moving claw
    must not count as held, and a claw that closes fully is empty."""
    arm = SimArm()
    r = pick.pick_to_tray(arm, DEFAULT_POSES, policy=StayPolicy(DEFAULT_POSES["gripper_closed"]))
    assert (r["result"], r["attempts"]) == ("missed", 2)
    assert r["seconds"] < pick.POLICY_TIME_LIMIT_S        # handed over when the claw closed, not at the limit
    assert arm.object == "floor"


def test_moving_claw_is_never_counted_as_held():
    """While the claw is still travelling, the policy keeps driving (no hand-over)."""
    arm = SimArm(grip_speed_s=30.0)          # 1 unit per frame: slow, but well above GRIP_STILL_DEG
    seen = []
    policy = StayPolicy(DEFAULT_POSES["gripper_closed"])
    step = policy.step
    policy.step = lambda state: (seen.append(state["gripper"]), step(state))[1]
    picker = pick._Picker(arm, DEFAULT_POSES, 0.0, 0.0, 45, None, lambda m: None)
    arm.pos["gripper"] = arm.goal["gripper"] = DEFAULT_POSES["gripper_open"]
    assert picker.policy_grasp(policy) is False
    # The claw passed through the 'holding' band for many frames without a hand-over,
    # and the policy drove until it stood still fully closed for 8 frames.
    assert sum(pick.claw_holding(g, 60.0, 2.0) for g in seen) > pick.POLICY_HOLD_FRAMES
    assert sum(g == DEFAULT_POSES["gripper_closed"] for g in seen) == pick.POLICY_HOLD_FRAMES + 1


@pytest.mark.parametrize("grip", [DEFAULT_POSES["gripper_open"], 30.0])    # never closes / parks mid-range
def test_policy_time_limit_ends_in_missed(grip):
    arm = SimArm()
    r = pick.pick_to_tray(arm, DEFAULT_POSES, policy=StayPolicy(grip))
    assert (r["result"], r["attempts"]) == ("missed", 2)
    assert r["seconds"] >= 2 * pick.POLICY_TIME_LIMIT_S                     # the limit is per attempt
    assert r["seconds"] < 2 * pick.POLICY_TIME_LIMIT_S + 15


class CostlyPolicy(ScriptedPolicy):
    """Each step takes `cost` seconds on the sim clock (camera read + inference)."""

    def __init__(self, arm, cost):
        super().__init__()
        self.arm, self.cost, self.times = arm, cost, []

    def step(self, state):
        self.times.append(self.arm.clock.now())
        self.arm.step(self.cost)
        self.arm.clock.t += self.cost
        return super().step(state)


@pytest.mark.parametrize("cost, period", [(0.0, DT), (0.012, DT), (0.05, 0.05)])
def test_policy_loop_is_paced_to_rate(cost, period):
    arm = SimArm()
    policy = CostlyPolicy(arm, cost)
    r = pick.pick_to_tray(arm, DEFAULT_POSES, policy=policy)
    assert r["result"] == "in_tray"
    gaps = np.diff(policy.times)
    assert len(gaps) > 10
    assert np.allclose(gaps, period, atol=1e-9), gaps


def test_policy_failure_holds_the_arm_and_raises():
    class Stalls(ScriptedPolicy):
        def step(self, state):
            if self.steps == 20:
                raise TimeoutError("camera latest frame is too old")
            return super().step(state)

    arm = SimArm()
    sent = []
    command = arm.command
    arm.command = lambda t: (sent.append(dict(t)), command(t))
    with pytest.raises(TimeoutError):
        pick.pick_to_tray(arm, DEFAULT_POSES, policy=Stalls())
    pos = arm.read()
    assert all(sent[-1][j] == pytest.approx(pos[j]) for j in poses_mod.ARM_JOINTS)


def _spy_on_hold(monkeypatch):
    calls = []
    original = pick._Picker._hold
    monkeypatch.setattr(pick._Picker, "_hold", lambda self: (calls.append(1), original(self)))
    return calls


def test_policy_failure_calls_hold(monkeypatch):
    class Stalls(ScriptedPolicy):
        def step(self, state):
            if self.steps == 20:
                raise TimeoutError("camera latest frame is too old")
            return super().step(state)

    holds = _spy_on_hold(monkeypatch)
    with pytest.raises(TimeoutError):
        pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=Stalls())
    assert holds == [1]


@pytest.mark.parametrize("key", ["shoulder_pan", "elbow_flex", "gripper"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.float32("nan")])
def test_policy_non_finite_target_holds_instead_of_creeping(monkeypatch, key, bad):
    class Broken(ScriptedPolicy):
        def step(self, state):
            action = super().step(state)
            return {**action, key: bad} if self.steps >= 20 else action

    arm = SimArm()
    sent = []
    command = arm.command
    arm.command = lambda t: (sent.append(dict(t)), command(t))
    holds = _spy_on_hold(monkeypatch)
    with pytest.raises(ValueError, match=key):
        pick.pick_to_tray(arm, DEFAULT_POSES, policy=Broken())
    assert holds == [1]
    assert all(np.isfinite(v) for t in sent for v in t.values())   # nothing non-finite reached the arm


class JumpyPolicy:
    """Asks for wild targets: the speed cap must still hold."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def step(self, state):
        return {j: float(self.rng.uniform(-150, 150)) for j in poses_mod.JOINTS}


def test_policy_targets_are_speed_capped():
    arm = SimArm()
    sent = []
    command = arm.command
    arm.command = lambda t: (sent.append(dict(t)), command(t))
    pick.pick_to_tray(arm, DEFAULT_POSES, policy=JumpyPolicy(), speed_deg_s=25)
    for a, b in zip(sent, sent[1:]):
        for j in poses_mod.ARM_JOINTS:
            assert abs(b[j] - a[j]) <= 25 / pick.RATE_HZ + 1e-6


# --- webcams: opened like T1 records them, closed on every path -----------------

class FakeLeRobotCamera:
    instances = []

    def __init__(self, config):
        self.config = config
        self.max_ages = []
        FakeLeRobotCamera.instances.append(self)

    def connect(self):
        pass

    def read_latest(self, max_age_ms=500):
        self.max_ages.append(max_age_ms)
        return np.zeros((self.config.height, self.config.width, 3), np.uint8)

    def read(self):
        raise AssertionError("read() blocks up to 10 s waiting for a new frame")

    def disconnect(self):
        pass


def test_opencv_frames_match_recording_settings_and_never_block(monkeypatch):
    pytest.importorskip("lerobot.cameras.opencv")
    import lerobot.cameras.opencv as lr_opencv

    FakeLeRobotCamera.instances = []
    monkeypatch.setattr(lr_opencv, "OpenCVCamera", FakeLeRobotCamera)
    cam = policy_mod.OpenCVFrames(1, 320, 180, 30)
    (fake,) = FakeLeRobotCamera.instances
    assert (fake.config.index_or_path, fake.config.width, fake.config.height, fake.config.fps) == (1, 320, 180, 30)
    assert cam.read().shape == (180, 320, 3)
    assert fake.max_ages == [policy_mod.FRAME_MAX_AGE_MS]
    cam.close()


class FakeCam:
    """Stands in for OpenCVFrames; records opens and closes."""
    log = []
    fail_on = None

    def __init__(self, index, width, height, fps):
        if index == FakeCam.fail_on:
            raise ConnectionError(f"camera {index} busy")
        self.args = (index, width, height, fps)
        FakeCam.log.append(("open", index, width, height, fps))

    def read(self):
        return np.zeros((self.args[2], self.args[1], 3), np.uint8)

    def close(self):
        FakeCam.log.append(("close", self.args[0]))


class FakeAct:
    """Stands in for ActGrasp (no torch): drives like ScriptedPolicy."""
    fail = False

    def __init__(self, path, cameras, device=None):
        if FakeAct.fail:
            raise RuntimeError("bad checkpoint")
        self.cameras = cameras
        self.inner = ScriptedPolicy()

    def reset(self):
        self.inner.reset()

    def step(self, state):
        return self.inner.step(state)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    FakeCam.log, FakeCam.fail_on, FakeAct.fail = [], None, False
    monkeypatch.setattr(policy_mod, "OpenCVFrames", FakeCam)
    monkeypatch.setattr(policy_mod, "ActGrasp", FakeAct)
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"type": "act", "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 180, 320]},
        "observation.images.scene": {"type": "VISUAL", "shape": [3, 180, 320]}}}))
    return ckpt


def opened():
    return [e for e in FakeCam.log if e[0] == "open"]


def closed():
    return [e for e in FakeCam.log if e[0] == "close"]


def test_load_policy_opens_robot_yaml_cameras_at_checkpoint_size(fakes, tmp_path):
    policy = pick.load_policy(fakes, config_path=write_config(tmp_path))
    assert sorted(opened()) == [("open", 1, 320, 180, pick.RATE_HZ), ("open", 2, 320, 180, pick.RATE_HZ)]
    assert set(policy.cameras) == {"wrist", "scene"}


def test_load_policy_capture_size(fakes, tmp_path):
    pick.load_policy(fakes, config_path=write_config(tmp_path), capture_size=(640, 360))
    assert sorted(opened()) == [("open", 1, 640, 360, pick.RATE_HZ), ("open", 2, 640, 360, pick.RATE_HZ)]


def test_load_policy_refuses_capture_size_that_would_squash(fakes, tmp_path):
    with pytest.raises(ValueError, match="aspect ratio"):
        pick.load_policy(fakes, config_path=write_config(tmp_path), capture_size=(640, 480))
    assert FakeCam.log == []


def test_load_policy_closes_cameras_when_the_checkpoint_fails(fakes, tmp_path):
    FakeAct.fail = True
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        pick.load_policy(fakes, config_path=write_config(tmp_path))
    assert len(opened()) == 2
    assert sorted(closed()) == [("close", 1), ("close", 2)]


def test_load_policy_closes_cameras_when_a_later_camera_fails(fakes, tmp_path):
    FakeCam.fail_on = 2
    with pytest.raises(ConnectionError):
        pick.load_policy(fakes, config_path=write_config(tmp_path))
    assert closed() == [("close", e[1]) for e in opened()]


@pytest.mark.parametrize("cameras, words", [({"wrist": 1}, "scene"), ({"wrist": 1, "scene": True}, "cameras.scene"),
                                            ({"wrist": 1, "scene": None}, "cameras.scene")])
def test_load_policy_needs_every_camera_in_robot_yaml(fakes, tmp_path, cameras, words):
    with pytest.raises(pick.ConfigError, match=words):
        pick.load_policy(fakes, config_path=write_config(tmp_path, cameras=cameras))
    assert FakeCam.log == []


def test_load_policy_without_robot_yaml_has_no_fallback(fakes, tmp_path):
    with pytest.raises(pick.ConfigError, match="not found"):
        pick.load_policy(fakes, config_path=tmp_path / "missing.yaml")
    assert FakeCam.log == []


def test_pick_to_tray_with_a_checkpoint_folder_closes_its_cameras(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(pick, "DEFAULT_CONFIG_PATH", write_config(tmp_path))
    r = pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=str(fakes))
    assert (r["result"], r["grasp"]) == ("in_tray", "policy")
    assert len(opened()) == 2 and len(closed()) == 2


def test_one_camera_failing_to_close_does_not_leave_the_other_open(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(pick, "DEFAULT_CONFIG_PATH", write_config(tmp_path))
    close = FakeCam.close

    def flaky_close(self):
        close(self)
        if self.args[0] == 1:
            raise OSError("device gone")

    monkeypatch.setattr(FakeCam, "close", flaky_close)
    r = pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=fakes)
    assert r["result"] == "in_tray"
    assert "camera wrist close failed: OSError: device gone" in r["detail"]
    assert sorted(closed()) == [("close", 1), ("close", 2)]


def test_pick_to_tray_closes_cameras_when_the_run_fails(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(pick, "DEFAULT_CONFIG_PATH", write_config(tmp_path))
    monkeypatch.setattr(FakeAct, "step", lambda self, state: (_ for _ in ()).throw(TimeoutError("stale frame")))
    with pytest.raises(TimeoutError):
        pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=fakes)
    assert len(opened()) == 2 and len(closed()) == 2


class FakeRealArm:
    """Stands in for RealArm."""
    fail = False

    def __init__(self, **kwargs):
        if FakeRealArm.fail:
            raise ConnectionError("no such port")
        self.sim = SimArm()
        self.clock = self.sim.clock

    def read(self):
        return self.sim.read()

    def command(self, targets):
        self.sim.command(targets)

    def close(self):
        pass


@pytest.mark.parametrize("arm_fails", [False, True])
def test_cli_real_policy_mode_closes_cameras(fakes, tmp_path, capsys, monkeypatch, arm_fails):
    FakeRealArm.fail = arm_fails
    monkeypatch.setattr(real_mod, "RealArm", FakeRealArm)
    poses_path = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, poses_path)
    rc = pick.main(["run", "--policy", str(fakes), "--poses", str(poses_path),
                    "--config", str(write_config(tmp_path)), "--speed", "25"])
    result = last_result(capsys)
    assert result["grasp"] == "policy"
    if arm_fails:
        assert (rc, result["result"]) == (1, "error")
        assert "no such port" in result["detail"]
    else:
        assert (rc, result["result"]) == (0, "taken")      # hand mode is the default (A7)
    assert len(opened()) == 2 and sorted(closed()) == [("close", 1), ("close", 2)]


class _Loaded(Exception):
    pass


@pytest.mark.parametrize("entry", [pick.pick_to_hand, pick.pick_to_tray])
@pytest.mark.parametrize("size", [None, (640, 360)])
def test_pick_passes_camera_size_to_load_policy(monkeypatch, tmp_path, entry, size):
    """A wrist webcam that can only record 640x360 must still run a 320x180 checkpoint."""
    seen = {}

    def fake_load(path, **kw):
        seen.update(kw)
        raise _Loaded

    monkeypatch.setattr(pick, "load_policy", fake_load)
    poses = dict(DEFAULT_POSES)
    with pytest.raises(_Loaded):
        entry(SimArm(), poses, policy=tmp_path, camera_size=size)
    assert seen.get("capture_size") == size


@pytest.mark.parametrize("entry", [pick.pick_to_hand, pick.pick_to_tray])
def test_camera_size_without_a_checkpoint_is_an_error(entry):
    with pytest.raises(ValueError, match="camera_size"):
        entry(SimArm(), dict(DEFAULT_POSES), policy=None, camera_size=(640, 360))
