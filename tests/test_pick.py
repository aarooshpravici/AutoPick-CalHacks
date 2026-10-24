"""Dry-run tests for the pick sequence (plan 10.1 "done when" 1). No hardware."""

import json
import threading

import pytest

from arm import pick, poses as poses_mod
from arm.pick import claw_holding, pick_to_tray
from arm.sim import DEFAULT_POSES, SimArm


def run(arm, **kwargs):
    return pick_to_tray(arm, DEFAULT_POSES, **kwargs)


def test_clean_pick():
    arm = SimArm()
    r = run(arm)
    assert r["result"] == "in_tray"
    assert r["grasp"] == "scripted"
    assert r["attempts"] == 1
    assert r["seconds"] > 0
    assert arm.object == "tray"


def test_one_miss_then_pick():
    arm = SimArm(miss_first=1)
    r = run(arm)
    assert r["result"] == "in_tray"
    assert r["attempts"] == 2
    assert arm.object == "tray"


def test_two_misses():
    arm = SimArm(miss_first=2)
    r = run(arm)
    assert r["result"] == "missed"
    assert r["attempts"] == 2
    assert arm.object == "floor"


def test_drop_while_lifting():
    arm = SimArm(slip_on_lift=True)
    r = run(arm)
    assert r["result"] == "dropped"
    assert arm.object == "floor"


def test_stuck_joint_holds():
    arm = SimArm(stall_joint="elbow_flex")
    r = run(arm)
    assert r["result"] == "stuck"
    assert "elbow_flex" in r["detail"]
    assert arm.goal["elbow_flex"] == arm.pos["elbow_flex"]   # holding where it is


def test_stop_event_holds():
    arm = SimArm()
    stop = threading.Event()
    ticks = {"n": 0}
    sleep = arm.clock.sleep

    def sleep_then_stop(dt):
        sleep(dt)
        ticks["n"] += 1
        if ticks["n"] == 60:
            stop.set()

    arm.clock.sleep = sleep_then_stop
    r = run(arm, stop_event=stop)
    assert r["result"] == "stopped"
    assert all(arm.goal[j] == arm.pos[j] for j in poses_mod.ARM_JOINTS)


class Recorder(SimArm):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent = []

    def command(self, targets):
        self.sent.append(dict(targets))
        super().command(targets)


def test_offsets_reach_pan_and_roll():
    arm = Recorder()
    assert run(arm, pan_deg=7.0, roll_deg=-3.0)["result"] == "in_tray"
    at = DEFAULT_POSES["at_object"]
    assert any(abs(c["shoulder_pan"] - (at["shoulder_pan"] + 7.0)) < 1e-6
               and abs(c["wrist_roll"] - (at["wrist_roll"] - 3.0)) < 1e-6 for c in arm.sent)


def test_claw_target_never_ramped():
    """Every command uses one of the three claw targets: open, closed-and-squeezing."""
    arm = Recorder()
    run(arm)
    g_open, g_closed = DEFAULT_POSES["gripper_open"], DEFAULT_POSES["gripper_closed"]
    assert {c["gripper"] for c in arm.sent} == {g_open, g_closed - pick.SQUEEZE}


def test_speed_limits_joint_steps():
    arm = Recorder()
    run(arm, speed_deg_s=25)
    for a, b in zip(arm.sent, arm.sent[1:]):
        for j in poses_mod.ARM_JOINTS:
            assert abs(b[j] - a[j]) <= 25 / pick.RATE_HZ + 1e-6


def test_rejects_unsafe_speed():
    with pytest.raises(ValueError):
        run(SimArm(), speed_deg_s=500)


def test_claw_holding():
    assert claw_holding(17, g_open=60, g_closed=2)
    assert not claw_holding(3, g_open=60, g_closed=2)        # fully closed: empty
    assert not claw_holding(60, g_open=60, g_closed=2)       # open
    assert claw_holding(80, g_open=40, g_closed=95)          # reversed scale


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


def test_policy_grasp():
    arm = SimArm()
    r = run(arm, policy=ScriptedPolicy())
    assert r["result"] == "in_tray"
    assert r["grasp"] == "policy"
    assert arm.object == "tray"


def test_poses_roundtrip(tmp_path):
    path = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, path)
    loaded = poses_mod.load(path)
    assert loaded["at_object"] == DEFAULT_POSES["at_object"]
    bad = dict(DEFAULT_POSES)
    del bad["lift"]
    with pytest.raises(poses_mod.PosesError):
        poses_mod.validate(bad)


@pytest.mark.parametrize("extra, result, code", [
    ([], "in_tray", 0),
    (["--sim-miss", "2"], "missed", 2),
    (["--sim-slip"], "dropped", 3),
    (["--sim-stall", "shoulder_lift"], "stuck", 4),
])
def test_cli_dry_run(tmp_path, capsys, extra, result, code):
    missing = tmp_path / "none.json"
    rc = pick.main(["run", "--dry-run", "--poses", str(missing), "--config", str(missing)] + extra)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("RESULT ")
    assert json.loads(last[len("RESULT "):])["result"] == result
    assert rc == code
