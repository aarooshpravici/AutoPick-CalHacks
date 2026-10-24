"""Dry-run tests for teaching poses (task A3). No hardware."""

import threading

from arm import pick, poses as poses_mod
from arm.sim import DEFAULT_POSES, SimArm, SimLeader
from arm.teach import STEPS, save_with_backup, teach


def scripted_person(leader, follower, targets):
    """For each step: move the leader to the target, press Enter once the follower is there."""
    names = iter(name for name, _ in STEPS)

    def wait_for(prompt):
        name = next(names)
        leader.pos = dict(targets[name])
        pressed = threading.Event()
        sleep = follower.clock.sleep

        def sleep_and_check(dt):
            sleep(dt)
            if all(abs(follower.pos[j] - leader.pos[j]) < 0.5 for j in poses_mod.JOINTS):
                pressed.set()
                follower.clock.sleep = sleep

        follower.clock.sleep = sleep_and_check
        return pressed

    return wait_for


def targets_from(poses):
    t = {name: poses[name] for name in poses_mod.POSE_NAMES}
    t["gripper_open"] = {**poses["above_tray"], "gripper": poses["gripper_open"]}
    t["gripper_closed"] = {**poses["above_tray"], "gripper": poses["gripper_closed"]}
    return t


def test_teach_records_every_pose_and_saves(tmp_path):
    leader, follower = SimLeader(), SimArm()
    taught = teach(leader, follower, speed_deg_s=45,
                   wait_for=scripted_person(leader, follower, targets_from(DEFAULT_POSES)), log=lambda m: None)
    for name in poses_mod.POSE_NAMES:
        for j in poses_mod.ARM_JOINTS:
            assert abs(taught[name][j] - DEFAULT_POSES[name][j]) < 0.5
    assert abs(taught["gripper_open"] - DEFAULT_POSES["gripper_open"]) < 0.5
    assert abs(taught["gripper_closed"] - DEFAULT_POSES["gripper_closed"]) < 0.5

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
    teach(leader, follower, speed_deg_s=25,
          wait_for=scripted_person(leader, follower, targets_from(DEFAULT_POSES)), log=lambda m: None)
    for a, b in zip(sent, sent[1:]):
        for j in poses_mod.ARM_JOINTS:
            assert abs(b[j] - a[j]) <= 25 / pick.RATE_HZ + 1e-6


def test_existing_poses_are_backed_up(tmp_path, monkeypatch):
    import arm.teach as teach_mod
    monkeypatch.setattr(teach_mod, "REPO_ROOT", tmp_path)
    out = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, out)
    backup = save_with_backup(DEFAULT_POSES, out)
    assert backup is not None and backup.exists()
