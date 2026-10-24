"""Tests for the ACT grasp mode (task T3) with a tiny untrained checkpoint. No hardware."""

import json

import numpy as np
import pytest

from arm import pick, poses as poses_mod
from arm.sim import DEFAULT_POSES, SimArm

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A small ACT checkpoint shaped like ours: 6-joint state/action, wrist camera 320x180."""
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


def test_act_step_gives_six_joint_targets(checkpoint):
    from arm.policy import ActGrasp, BlankFrames

    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames(640, 480)}, device="cpu")   # resized to 320x180
    policy.reset()
    action = policy.step(dict(DEFAULT_POSES["ready"]))
    assert set(action) == set(poses_mod.JOINTS)
    assert all(np.isfinite(v) for v in action.values())


def test_missing_camera_is_an_error(checkpoint):
    from arm.policy import ActGrasp

    with pytest.raises(ValueError, match="wrist"):
        ActGrasp(checkpoint, cameras={}, device="cpu")


def test_untrained_policy_runs_a_whole_pick(checkpoint):
    from arm.policy import ActGrasp, BlankFrames

    policy = ActGrasp(checkpoint, cameras={"wrist": BlankFrames()}, device="cpu")
    r = pick.pick_to_tray(SimArm(), DEFAULT_POSES, policy=policy)
    assert r["grasp"] == "policy"
    assert r["result"] in {"in_tray", "missed", "dropped", "stuck"}


def test_cli_dry_run_with_checkpoint(checkpoint, tmp_path, capsys):
    missing = tmp_path / "none"
    rc = pick.main(["run", "--dry-run", "--policy", str(checkpoint), "--poses", str(missing), "--config", str(missing)])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RESULT "))
    assert result["grasp"] == "policy"
    assert result["result"] != "error", result["detail"]
    assert rc == pick.EXIT_CODES[result["result"]]


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
