"""Recording command tests (task T1). The command is checked with lerobot-record's own
config parser; nothing is recorded and no hardware is opened."""

import pytest

from training import record

CFG = {
    "arm": {"follower_port": "/dev/tty.usbmodemF", "leader_port": "/dev/tty.usbmodemL",
            "follower_id": "floorpicker_follower", "leader_id": "floorpicker_leader",
            "max_relative_target_deg": 10},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
}


def build(**kw):
    args = dict(name="rehearsal", episodes=5, cameras=["wrist"], width=320, height=180, fps=30,
                episode_time_s=20, reset_time_s=10, resume=False, exe="lerobot-record")
    args.update(kw)
    return record.build_command(CFG, **args)


def parse_with_lerobot(cmd):
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    return draccus.parse(RecordConfig, args=cmd[1:])


def test_lerobot_accepts_the_command():
    cfg = parse_with_lerobot(build())
    assert cfg.robot.port == "/dev/tty.usbmodemF" and cfg.robot.id == "floorpicker_follower"
    assert cfg.teleop.port == "/dev/tty.usbmodemL"
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


def test_print_only(tmp_path, capsys):
    cfg = tmp_path / "robot.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(CFG))
    assert record.main(["--name", "demo1", "--episodes", "3", "--config", str(cfg), "--print-only"]) == 0
    out = capsys.readouterr().out
    assert "--dataset.num_episodes=3" in out and "local/demo1" in out


def test_bad_name():
    with pytest.raises(SystemExit):
        record.main(["--name", "../x", "--print-only"])
