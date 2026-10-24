"""Dry-run tests for the pick sequence (plan 10.1 "done when" 1). No hardware."""

import json
import math
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

import arm.real as real_mod
import arm.sim as sim_mod
from arm import pick, poses as poses_mod
from arm.pick import claw_holding, pick_to_tray
from arm.sim import DEFAULT_POSES, SimArm

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
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


def run_cli(tmp_path, capsys, *extra, config=None):
    """Run `python -m arm.pick run ...` in-process; returns (exit code, RESULT dict, stdout)."""
    config = config or write_config(tmp_path)
    argv = ["run", "--poses", str(tmp_path / "none.json"), "--config", str(config), *extra]
    rc = pick.main(argv)
    out = capsys.readouterr().out
    last = out.strip().splitlines()[-1]
    assert last.startswith("RESULT ")
    return rc, json.loads(last[len("RESULT "):]), out


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


@pytest.mark.parametrize("speed", [500, 90, 45.5, 0, -10, math.nan, math.inf, True, "25"])
def test_rejects_unsafe_speed(speed):
    """pick_to_tray never runs above the plan's 45 deg/s arm limit (section 17)."""
    assert pick.MAX_SPEED_DEG_S == 45
    with pytest.raises(ValueError):
        run(SimArm(), speed_deg_s=speed)


@pytest.mark.parametrize("bad", [{"pan_deg": math.nan}, {"roll_deg": math.inf}, {"pan_deg": None}])
def test_rejects_non_finite_offsets(bad):
    with pytest.raises(ValueError):
        run(SimArm(), **bad)


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
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", *extra)
    assert r["result"] == result
    assert rc == code


# --- speed ceiling from robot.yaml (review: hard-coded 90 deg/s cap) ---------------

def test_resolve_speed():
    cfg = {"move_speed_deg_s": 25}
    assert pick.resolve_speed(None, cfg) == 25          # default: the robot.yaml value
    assert pick.resolve_speed(25, cfg) == 25
    assert pick.resolve_speed(10.5, cfg) == 10.5
    for bad in (25.1, 45, 90, 0, -1, math.nan, math.inf):
        with pytest.raises(ValueError):
            pick.resolve_speed(bad, cfg)


def capture_speed(monkeypatch):
    seen = []
    real = pick.pick_to_tray

    def spy(*args, **kwargs):
        seen.append(kwargs["speed_deg_s"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pick, "pick_to_tray", spy)
    return seen


@pytest.mark.parametrize("ceiling, extra, expected", [
    (45, [], 45),
    (25, [], 25),
    (25, ["--speed", "25"], 25),
    (45, ["--speed", "25"], 25),
    (10, ["--speed", "4"], 4),
])
def test_cli_speed_defaults_to_and_stays_under_config(tmp_path, capsys, monkeypatch, ceiling, extra, expected):
    seen = capture_speed(monkeypatch)
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", *extra,
                       config=write_config(tmp_path, move_speed_deg_s=ceiling))
    assert (rc, r["result"]) == (0, "in_tray")
    assert seen == [expected]


class NoHardware:
    """Stands in for RealArm; records every attempt to open the real arm."""
    opened = []

    def __init__(self, **kwargs):
        NoHardware.opened.append(kwargs)
        self.sim = SimArm()
        self.clock = self.sim.clock

    def read(self):
        return self.sim.read()

    def command(self, targets):
        self.sim.command(targets)

    def close(self):
        pass


@pytest.fixture
def fake_real_arm(monkeypatch, tmp_path):
    NoHardware.opened = []
    monkeypatch.setattr(real_mod, "RealArm", NoHardware)
    poses_path = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, poses_path)
    return poses_path


@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_refuses_speed_above_config(tmp_path, capsys, monkeypatch, fake_real_arm, mode):
    seen = capture_speed(monkeypatch)
    extra = ["--dry-run"] if mode == "dry" else []
    rc, r, _ = run_cli(tmp_path, capsys, "--speed", "90", "--poses", str(fake_real_arm), *extra,
                       config=write_config(tmp_path, move_speed_deg_s=10))
    assert (rc, r["result"]) == (1, "error")
    assert "move_speed_deg_s" in r["detail"] and "90" in r["detail"] and "10" in r["detail"]
    assert seen == []                 # the pick never started
    assert NoHardware.opened == []    # and the arm was never opened


def test_cli_real_mode_reads_arm_settings_from_config(tmp_path, capsys, fake_real_arm):
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm),
                       config=write_config(tmp_path, max_relative_target_deg=7))
    assert (rc, r["result"]) == (0, "in_tray")
    assert NoHardware.opened == [{"port": "/dev/tty.usbmodemTEST1", "robot_id": "so101_follower_arm",
                                  "max_relative_target": 7}]


# --- robot.yaml validation (review: unhelpful or unsafe config handling) ----------

@pytest.mark.parametrize("overrides, words", [
    ({"follower_port": DROP}, "arm.follower_port"),
    ({"follower_id": ""}, "arm.follower_id"),
    ({"max_relative_target_deg": None}, "arm.max_relative_target_deg"),   # None = no step cap
    ({"max_relative_target_deg": 0}, "arm.max_relative_target_deg"),
    ({"max_relative_target_deg": DROP}, "arm.max_relative_target_deg"),
    ({"move_speed_deg_s": DROP}, "arm.move_speed_deg_s"),
    ({"move_speed_deg_s": None}, "arm.move_speed_deg_s"),
    ({"move_speed_deg_s": "fast"}, "arm.move_speed_deg_s"),
    ({"move_speed_deg_s": True}, "arm.move_speed_deg_s"),
    ({"move_speed_deg_s": -5}, "arm.move_speed_deg_s"),
    ({"move_speed_deg_s": 60}, "45 deg/s arm limit"),
])
def test_load_arm_config_rejects_bad_values(tmp_path, overrides, words):
    with pytest.raises(pick.ConfigError, match=words.replace(".", r"\.")):
        pick.load_arm_config(write_config(tmp_path, **overrides))


@pytest.mark.parametrize("text", ["", "arm:\n", "arm: 5\n", "cameras: {wrist: 1}\n"])
def test_load_arm_config_rejects_missing_arm_section(tmp_path, text):
    path = tmp_path / "robot.yaml"
    path.write_text(text)
    with pytest.raises(pick.ConfigError, match="'arm:' section"):
        pick.load_arm_config(path)


def test_load_arm_config_missing_file_has_no_fallback(tmp_path):
    with pytest.raises(pick.ConfigError, match="not found"):
        pick.load_arm_config(tmp_path / "none.yaml")


def test_load_arm_config_ok(tmp_path):
    cfg = pick.load_arm_config(write_config(tmp_path))
    assert cfg["follower_id"] == "so101_follower_arm"
    assert cfg["move_speed_deg_s"] == 45
    with pytest.raises(pick.ConfigError, match="arm.leader_port"):
        pick.load_arm_config(write_config(tmp_path, leader_port=DROP), require=("leader_port",))


@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_bad_config_is_a_clear_error(tmp_path, capsys, fake_real_arm, mode):
    extra = ["--dry-run"] if mode == "dry" else []
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm), *extra,
                       config=write_config(tmp_path, max_relative_target_deg=None))
    assert (rc, r["result"]) == (1, "error")
    assert r["detail"].startswith("ConfigError:") and "max_relative_target_deg" in r["detail"]
    assert NoHardware.opened == []


def test_cli_missing_config_is_an_error(tmp_path, capsys):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", config=tmp_path / "none.yaml")
    assert (rc, r["result"]) == (1, "error")
    assert "not found" in r["detail"]


def test_sim_rejects_unknown_stall_joint():
    with pytest.raises(ValueError, match="stall_joint"):
        SimArm(stall_joint="knee")


@pytest.mark.parametrize("argv", [["run", "--dry-run", "--sim-stall", "knee"], [], ["run", "--pan", "left"]])
def test_cli_bad_arguments_end_in_result_and_exit_1(tmp_path, capsys, argv):
    """argparse's own exit code 2 would read as 'missed'."""
    rc = pick.main(argv + (["--config", str(write_config(tmp_path))] if argv else []))
    captured = capsys.readouterr()
    last = captured.out.strip().splitlines()[-1]
    assert rc == 1
    assert json.loads(last[len("RESULT "):])["result"] == "error"
    assert "usage" in captured.err


# --- end of the command line (review: close errors, signal handlers) --------------

def test_cli_close_failure_still_prints_result(tmp_path, capsys, monkeypatch):
    class BadClose(SimArm):
        def close(self):
            raise OSError("port vanished")

    monkeypatch.setattr(sim_mod, "SimArm", BadClose)
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run")
    assert (rc, r["result"]) == (0, "in_tray")
    assert "close failed: OSError: port vanished" in r["detail"]


def test_cli_interrupt_before_pick_still_prints_result(tmp_path, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sim_mod, "SimArm", interrupted)
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run")
    assert (rc, r["result"]) == (1, "stopped")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_stops_pick_and_handler_is_restored(tmp_path, capsys, monkeypatch, sig):
    class SignalMidPick(SimArm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sleep, ticks = self.clock.sleep, [0]

            def sleep_then_signal(dt):
                sleep(dt)
                ticks[0] += 1
                if ticks[0] == 30:
                    signal.raise_signal(sig)

            self.clock.sleep = sleep_then_signal

    caught = []
    previous = signal.signal(sig, lambda *a: caught.append(a))   # stand-in for pytest's own handler
    try:
        ours = signal.getsignal(sig)
        monkeypatch.setattr(sim_mod, "SimArm", SignalMidPick)
        rc, r, _ = run_cli(tmp_path, capsys, "--dry-run")
        assert (rc, r["result"]) == (1, "stopped")
        assert caught == []                       # main's handler took the signal
        assert signal.getsignal(sig) is ours      # and put ours back afterwards
    finally:
        signal.signal(sig, previous)


def test_cli_off_main_thread_still_prints_result(tmp_path, capsys):
    config = write_config(tmp_path)
    before = signal.getsignal(signal.SIGINT)
    out = {}
    t = threading.Thread(target=lambda: out.update(rc=pick.main(
        ["run", "--dry-run", "--poses", str(tmp_path / "none.json"), "--config", str(config)])))
    t.start()
    t.join(timeout=30)
    assert out.get("rc") == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("RESULT ")
    assert signal.getsignal(signal.SIGINT) is before


# --- arrival tolerance and the 2 s stuck rule (plan 10.1) ------------------------

class SagArm(SimArm):
    """One joint reads `sag` degrees short of where it was sent, like a soft servo under load."""

    def __init__(self, joint, sag, **kw):
        super().__init__(**kw)
        self.sag_joint, self.sag = joint, sag

    def read(self):
        pos = super().read()
        pos[self.sag_joint] -= self.sag
        return pos


@pytest.mark.parametrize("joint", sorted(pick.TOLERANCE_DEG))
def test_joint_settling_inside_tolerance_arrives(joint):
    arm = SagArm(joint, pick.TOLERANCE_DEG[joint] - 0.5)
    r = run(arm)
    assert r["result"] == "in_tray", r["detail"]
    assert arm.object == "tray"


@pytest.mark.parametrize("joint", sorted(pick.TOLERANCE_DEG))
def test_joint_settling_outside_tolerance_is_stuck_after_2s(joint):
    sag = pick.TOLERANCE_DEG[joint] + 0.5
    r = run(SagArm(joint, sag), speed_deg_s=45)
    assert r["result"] == "stuck"
    assert r["detail"].startswith("ready:") and joint in r["detail"]
    # The first move (to ready) starts `sag` deg away, so it should end after sag / speed
    # and be reported stuck 2 s later.
    # "seconds" is rounded to 0.01 s.
    stuck_at = sag / 45 + pick.STUCK_AFTER_S
    assert stuck_at - 0.01 <= r["seconds"] <= stuck_at + 2 / pick.RATE_HZ


def test_drop_between_lift_and_tray():
    class SlipOverTray(SimArm):
        def step(self, dt):
            super().step(dt)
            if self.object == "held" and self._near("above_tray"):
                self.object = "floor"                      # fell out on the way to the tray
                self._missing = True
                self.pos["gripper"] = self.goal["gripper"]  # claw snaps shut on nothing

    arm = SlipOverTray()
    r = run(arm)
    assert r["result"] == "dropped"
    assert "above_tray" in r["detail"]
    assert arm.object == "floor"


# --- repo-root conftest.py (review: bare pytest could not import the packages) ---

def test_bare_pytest_finds_the_packages():
    """`pytest` (not `python -m pytest`) doesn't put the repo root on sys.path; the empty
    repo-root conftest.py does. -I keeps the current directory and PYTHONPATH out."""
    assert (REPO_ROOT / "conftest.py").is_file()
    code = ("import sys, pytest; sys.exit(pytest.main(['-q', '--collect-only', "
            "'-p', 'no:cacheprovider', 'tests/test_pick.py']))")
    proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
