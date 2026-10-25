"""arm/teleop.py: safe teleop (frozen wrist, line-up first, speed and lead caps). No hardware."""

import logging
import signal

import pytest
import yaml

import arm.real as real_mod
from arm import align as align_mod
from arm import teleop as teleop_mod
from arm.pick import RATE_HZ
from arm.poses import ARM_JOINTS
from arm.sim import DEFAULT_POSES, SimArm, SimLeader

READY = {**DEFAULT_POSES["ready"], "gripper": DEFAULT_POSES["gripper_open"]}


def clamp_warning(**joints):
    """LeRobot's step-cap warning, worded as lerobot/robots/utils.py logs it."""
    from pprint import pformat
    return ("Relative goal position magnitude had to be clamped to be safe.\n"
            + pformat({j: {"original goal_pos": v, "safe goal_pos": v / 2} for j, v in joints.items()}, indent=4))


class Recording(SimArm):
    def __init__(self, frozen=(), **kw):
        super().__init__(**kw)
        self.frozen = tuple(frozen)
        self.commands = []

    def command(self, targets):
        self.commands.append(dict(targets))
        super().command({**self.read(), **targets})


def test_follow_caps_each_step_and_the_lead():
    cmd = dict(READY)
    target = {**READY, "elbow_flex": READY["elbow_flex"] + 90, "shoulder_pan": READY["shoulder_pan"] - 90}
    teleop_mod.follow(cmd, target, READY, max_step=1.5)
    assert cmd["elbow_flex"] == pytest.approx(READY["elbow_flex"] + 1.5)
    assert cmd["shoulder_pan"] == pytest.approx(READY["shoulder_pan"] - 1.5)
    stuck = dict(cmd)                                   # the follower hasn't moved at all
    for _ in range(100):
        teleop_mod.follow(stuck, target, READY, max_step=1.5)
    assert stuck["elbow_flex"] == pytest.approx(READY["elbow_flex"] + teleop_mod.LEAD_DEG)
    assert stuck["gripper"] == target["gripper"]


def test_teleop_is_speed_capped_and_never_commands_a_frozen_joint():
    follower = Recording(frozen=("wrist_roll",))
    leader = SimLeader({**READY, "elbow_flex": READY["elbow_flex"] + 40, "wrist_roll": READY["wrist_roll"] + 90})
    teleop_mod.teleop(leader, follower, 30, max_s=2.0)
    max_step = 30 / RATE_HZ
    assert follower.commands and all("wrist_roll" not in c for c in follower.commands)
    for a, b in zip(follower.commands, follower.commands[1:-1]):
        assert abs(b["elbow_flex"] - a["elbow_flex"]) <= max_step + 1e-9
    assert follower.read()["elbow_flex"] > READY["elbow_flex"] + 20         # it really followed


def test_a_blocked_joint_does_not_wind_up():
    follower = Recording(stall_joint="elbow_flex")
    leader = SimLeader({**READY, "elbow_flex": READY["elbow_flex"] + 60})
    teleop_mod.teleop(leader, follower, 45, max_s=3.0)
    stuck = follower.read()["elbow_flex"]
    assert all(abs(c["elbow_flex"] - stuck) <= teleop_mod.LEAD_DEG + 1e-9 for c in follower.commands)


def test_stop_sends_nothing_more_then_holds_with_the_claw_kept():
    follower = Recording()
    leader = SimLeader({**READY, "gripper": DEFAULT_POSES["gripper_closed"], "elbow_flex": READY["elbow_flex"] + 30})
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return len(follower.commands) >= 5

    teleop_mod.teleop(leader, follower, 45, stop=stop)
    assert len(follower.commands) == 6                  # 5 follows, then exactly one hold
    hold, last_follow = follower.commands[-1], follower.commands[-2]
    assert hold["gripper"] == last_follow["gripper"]     # the claw keeps its grip
    for j in ARM_JOINTS:
        assert hold[j] == pytest.approx(follower.read()[j])


def test_claw_clamp_warning_is_muted_only_while_teleoperating(caplog):
    lerobot_log = logging.getLogger(teleop_mod.CLAMP_LOGGER)

    class Warns(Recording):
        def command(self, targets):
            lerobot_log.warning(clamp_warning(gripper=95.0))
            lerobot_log.warning(clamp_warning(elbow_flex=40.0))
            super().command(targets)

    with caplog.at_level("WARNING", logger=teleop_mod.CLAMP_LOGGER):
        teleop_mod.teleop(SimLeader(READY), Warns(), 45, max_s=0.2)
        during = [r.getMessage() for r in caplog.records]
        caplog.clear()
        lerobot_log.warning(clamp_warning(gripper=95.0))
    assert during and all("'elbow_flex'" in m for m in during)
    assert len(caplog.records) == 1 and lerobot_log.filters == []


def test_ctrl_c_only_counts_then_a_second_press_raises():
    presses = []
    restore = teleop_mod._catch_ctrl_c(presses)
    try:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert presses == [1]
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        restore()
    assert signal.getsignal(signal.SIGINT) is not handler


# --- command line, against fake hardware ------------------------------------------------

def write_config(tmp_path, **arm):
    cfg = {"arm": {"follower_port": "/dev/tty.TEST1", "leader_port": "/dev/tty.TEST2",
                   "follower_id": "f", "leader_id": "l", "max_relative_target_deg": 10,
                   "move_speed_deg_s": 45, "frozen_joints": ["wrist_roll"], **arm}}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def hw(monkeypatch):
    log = {"order": [], "opened": [], "closed": []}

    class Leader(SimLeader):
        def __init__(self, port, robot_id):
            if log.get("leader_fails"):
                raise RuntimeError("no matching calibration for the leader")
            super().__init__(READY)
            log["opened"].append("leader")
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.reads > 8:
                raise KeyboardInterrupt        # the person presses Ctrl-C (default handler in tests)
            return super().read()

        def close(self):
            log["closed"].append("leader")

    class Arm(Recording):
        def __init__(self, port, robot_id, max_relative_target, frozen=()):
            super().__init__(frozen=frozen)
            log["opened"].append(("follower", tuple(frozen)))
            log["arm"] = self

        def command(self, targets):
            log["order"].append("command")
            super().command(targets)

        def close(self):
            log["closed"].append("follower")

    def aligner(leader, follower):
        log["order"].append("align")
        align_mod.scripted_align(leader, follower)

    monkeypatch.setattr(real_mod, "RealLeader", Leader)
    monkeypatch.setattr(real_mod, "RealArm", Arm)
    monkeypatch.setattr(align_mod, "live_align", aligner)
    monkeypatch.setattr(teleop_mod, "_catch_ctrl_c", lambda presses: (lambda: None))
    return log


def test_cli_lines_up_before_the_first_command_and_closes_both(tmp_path, hw):
    assert teleop_mod.main(["--config", str(write_config(tmp_path)), "--speed", "25"]) == 0
    assert hw["opened"] == ["leader", ("follower", ("wrist_roll",))]
    assert hw["order"][0] == "align" and hw["order"].count("align") == 1 and "command" in hw["order"]
    assert all("wrist_roll" not in c for c in hw["arm"].commands)
    assert sorted(hw["closed"]) == ["follower", "leader"]


def test_cli_leader_failure_never_opens_the_follower(tmp_path, hw, capsys):
    hw["leader_fails"] = True
    assert teleop_mod.main(["--config", str(write_config(tmp_path))]) == 1
    assert hw["opened"] == [] and "no matching calibration" in capsys.readouterr().err


def test_cli_refuses_a_speed_above_the_config_before_opening_anything(tmp_path, hw, capsys):
    assert teleop_mod.main(["--config", str(write_config(tmp_path)), "--speed", "90"]) == 1
    assert hw["opened"] == []


def test_cli_line_up_stopped_means_nothing_moved(tmp_path, hw, monkeypatch, capsys):
    def stopped(leader, follower):
        raise align_mod.AlignError("input closed before Enter was pressed; nothing moved")

    monkeypatch.setattr(align_mod, "live_align", stopped)
    assert teleop_mod.main(["--config", str(write_config(tmp_path))]) == 1
    assert hw["arm"].commands == [] and "not started" in capsys.readouterr().err


def test_cli_dry_run_touches_no_hardware(tmp_path, hw, capsys):
    assert teleop_mod.main(["--config", str(write_config(tmp_path)), "--dry-run"]) == 0
    assert hw["opened"] == [] and "no hardware touched" in capsys.readouterr().out
