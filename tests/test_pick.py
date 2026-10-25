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
    "handoff": {"timeout_s": 60, "grip": "squeeze"},
}
DROP = object()     # write_config(...=DROP) leaves the key out


def write_config(tmp_path, handoff=None, **arm_overrides):
    """handoff: replaces the handoff section (DROP leaves it out)."""
    data = json.loads(json.dumps(ROBOT_YAML))
    if handoff is DROP:
        del data["handoff"]
    elif handoff is not None:
        data["handoff"] = handoff
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
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", "--to", "tray", *extra)
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
    """Speeds passed to pick_to_hand or pick_to_tray by the command line."""
    seen = []
    for name in ("pick_to_hand", "pick_to_tray"):
        def spy(*args, _real=getattr(pick, name), **kwargs):
            seen.append(kwargs["speed_deg_s"])
            return _real(*args, **kwargs)

        monkeypatch.setattr(pick, name, spy)
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
    assert (rc, r["result"]) == (0, "taken")       # hand mode is the default (A7)
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


@pytest.mark.parametrize("flag", ["--pan", "--roll"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_refuses_non_finite_offset_before_opening_the_arm(tmp_path, capsys, monkeypatch,
                                                               fake_real_arm, flag, value):
    seen = capture_speed(monkeypatch)
    rc, r, _ = run_cli(tmp_path, capsys, flag, value, "--poses", str(fake_real_arm),
                       config=write_config(tmp_path))
    assert (rc, r["result"]) == (1, "error")
    assert flag in r["detail"]
    assert seen == []                 # the pick never started
    assert NoHardware.opened == []    # and the arm was never opened (torque stays off)


def test_cli_real_mode_reads_arm_settings_from_config(tmp_path, capsys, fake_real_arm):
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm),
                       config=write_config(tmp_path, max_relative_target_deg=7))
    assert (rc, r["result"]) == (0, "taken")
    assert NoHardware.opened == [{"port": "/dev/tty.usbmodemTEST1", "robot_id": "so101_follower_arm",
                                  "max_relative_target": 7, "frozen": []}]


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
    assert (rc, r["result"]) == (0, "taken")
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


# --- A7: hand-off (plan 9.4, 10.1, 10.6, 17) ----------------------------------------

SQUEEZE_TARGET = DEFAULT_POSES["gripper_closed"] - pick.SQUEEZE


def hand(arm, **kwargs):
    return pick.pick_to_hand(arm, DEFAULT_POSES, **kwargs)


def where(arm):
    """The DEFAULT_POSES pose the arm is at (within the arrival tolerance), or None."""
    for name in poses_mod.POSE_NAMES:
        if all(abs(arm.pos[j] - DEFAULT_POSES[name][j]) <= pick.TOLERANCE_DEG[j] for j in poses_mod.ARM_JOINTS):
            return name
    return None


def holding_still(arm):
    """Every arm joint is commanded to stay where it is, and the claw still squeezes."""
    return (all(arm.goal[j] == arm.pos[j] for j in poses_mod.ARM_JOINTS)
            and arm.goal["gripper"] == SQUEEZE_TARGET)


def stop_when(arm, condition, after_ticks=0):
    """A stop_event set after_ticks sleeps after condition() first holds (checked after each
    sleep); stop.at is the clock then."""
    stop = threading.Event()
    ticks = {"n": None}
    sleep = arm.clock.sleep

    def sleep_then_stop(dt):
        sleep(dt)
        if stop.is_set():
            return
        if ticks["n"] is None:
            if not condition():
                return
            ticks["n"] = 0
        ticks["n"] += 1
        if ticks["n"] >= after_ticks:
            stop.at = arm.clock.now()
            stop.set()

    arm.clock.sleep = sleep_then_stop
    return stop


def watch_states(arm, stop_state=None, stop_after_ticks=None):
    """An on_state that records (state, object, pose the arm is at, clock, commands sent so far).
    With stop_state, the returned stop_event is set stop_after_ticks sleeps after that state
    is reported, and stop.at is the clock then."""
    seen = []

    def on_state(state):
        seen.append((state, arm.object, where(arm), arm.clock.now(), len(getattr(arm, "sent", []))))

    stop = stop_when(arm, lambda: stop_state in [s[0] for s in seen], stop_after_ticks or 0)
    return on_state, seen, stop


def test_handoff_taken_with_states():
    arm = Recorder(take_after_s=2.0)
    on_state, seen, _ = watch_states(arm)
    r = hand(arm, on_state=on_state)
    assert (r["result"], r["grasp"], r["attempts"]) == ("taken", "scripted", 1), r["detail"]
    # "pick" at the start, "present" as the move from lift starts, "wait_for_take" at present, holding
    assert [s[:3] for s in seen] == [("pick", "floor", "ready"), ("present", "held", "lift"),
                                     ("wait_for_take", "held", "present")]
    assert arm.object == "hand"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]
    assert r["seconds"] >= seen[-1][3] + 2.0         # the person took it 2 s after it was presented
    # while waiting it keeps squeezing: the claw target is only ever open or squeezing
    assert {c["gripper"] for c in arm.sent} == {DEFAULT_POSES["gripper_open"], SQUEEZE_TARGET}


def test_handoff_timeout_ends_at_lift_still_holding():
    arm = Recorder(never_take=True)
    on_state, seen, _ = watch_states(arm)
    r = hand(arm, handoff_timeout_s=5, on_state=on_state)
    assert r["result"] == "timeout"
    assert "5 s" in r["detail"]
    assert [s[0] for s in seen] == ["pick", "present", "wait_for_take"]
    assert arm.clock.now() - seen[-1][3] >= 5.0       # held at present for the whole timeout
    # still squeezing: the carry target is re-sent at every read, about 20 Hz
    present = {j: DEFAULT_POSES["present"][j] for j in poses_mod.ARM_JOINTS}
    waiting = [c for c in arm.sent[seen[-1][4]:]
               if c["gripper"] == SQUEEZE_TARGET and all(c[j] == v for j, v in present.items())]
    assert len(waiting) >= 5 * pick.TAKE_POLL_HZ
    assert where(arm) == "lift"
    assert arm.object == "held" and claw_holding(arm.pos["gripper"], DEFAULT_POSES["gripper_open"],
                                                 DEFAULT_POSES["gripper_closed"])
    assert arm.sent[-1]["gripper"] == SQUEEZE_TARGET


def test_handoff_timeout_default_is_60_s():
    arm = SimArm(never_take=True)
    on_state, seen, _ = watch_states(arm)
    assert hand(arm, on_state=on_state)["result"] == "timeout"
    assert 60.0 <= arm.clock.now() - seen[-1][3] <= 70.0


def test_stop_while_waiting_holds_at_present():
    arm = SimArm(never_take=True)
    on_state, seen, stop = watch_states(arm, "wait_for_take", 10)
    r = hand(arm, on_state=on_state, stop_event=stop)
    assert r["result"] == "stopped"
    assert arm.clock.now() - stop.at <= 0.5            # plan 17: the arm holds within 0.5 s
    assert [s[0] for s in seen] == ["pick", "present", "wait_for_take"]
    assert where(arm) == "present" and arm.object == "held"
    assert holding_still(arm)


def test_stop_while_moving_to_present_holds_where_it_is():
    arm = SimArm(never_take=True)
    on_state, seen, stop = watch_states(arm, "present", 20)
    r = hand(arm, on_state=on_state, stop_event=stop)
    assert r["result"] == "stopped"
    assert arm.clock.now() - stop.at <= 0.5
    assert [s[0] for s in seen] == ["pick", "present"]
    assert where(arm) is None and arm.object == "held"   # between lift and present
    assert holding_still(arm)


class SlipOnTheWayToPresent(SimArm):
    """The object falls out halfway from lift to present."""

    been_at_lift = False

    def step(self, dt):
        super().step(dt)
        if self.object == "held" and self._near("lift"):
            self.been_at_lift = True
        halfway = (self.poses["lift"]["elbow_flex"] + self.poses["present"]["elbow_flex"]) / 2
        if self.object == "held" and self.been_at_lift and self.pos["elbow_flex"] <= halfway:
            self.object = "floor"
            self._missing = True
            self.pos["gripper"] = self.goal["gripper"]   # claw snaps shut on nothing


def test_drop_on_the_way_to_present():
    arm = SlipOnTheWayToPresent()
    on_state, seen, _ = watch_states(arm)
    r = hand(arm, on_state=on_state)
    assert r["result"] == "dropped"
    assert "present" in r["detail"]
    assert [s[0] for s in seen] == ["pick", "present"]
    assert arm.object == "floor"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]


@pytest.mark.parametrize("speed, cap", [(45, 25), (25, 25), (10, 10)])
@pytest.mark.parametrize("taker, expected", [({}, "taken"), ({"never_take": True}, "timeout"),
                                             ({"never_take": True, "slip_at_present_after_s": 0.5}, "dropped")])
def test_moves_near_the_person_never_exceed_25_deg_s(speed, cap, taker, expected):
    """From the start of the move to present to the end (waiting, then back to ready or lift)."""
    arm = Recorder(**taker)
    marks = {}
    r = hand(arm, speed_deg_s=speed, handoff_timeout_s=3, on_state=lambda s: marks.setdefault(s, len(arm.sent)))
    assert r["result"] == expected

    def steps(sent):
        return [max(abs(b[j] - a[j]) for j in poses_mod.ARM_JOINTS) for a, b in zip(sent, sent[1:])]

    near_person = steps(arm.sent[marks["present"] - 1:])
    assert max(near_person) <= cap / pick.RATE_HZ + 1e-6
    assert max(near_person) >= cap / pick.RATE_HZ - 1e-6          # and it did move at that speed
    assert max(steps(arm.sent[:marks["present"]])) == pytest.approx(speed / pick.RATE_HZ)  # full speed elsewhere


def test_policy_grasp_then_handoff():
    arm = SimArm()
    on_state, seen, _ = watch_states(arm)
    r = hand(arm, policy=ScriptedPolicy(), on_state=on_state)
    assert (r["result"], r["grasp"], r["attempts"]) == ("taken", "policy", 1), r["detail"]
    assert [s[0] for s in seen] == ["pick", "present", "wait_for_take"]
    assert arm.object == "hand" and where(arm) == "ready"


class GlitchyClaw(SimArm):
    """Nobody takes it. While waiting for the take, the claw reads `value` ("closed" or "open")
    for `run` reads, then twice truly, over and over (run=None: for good)."""

    def __init__(self, value, run, **kw):
        super().__init__(never_take=True, **kw)
        self.value = self.poses["gripper_" + value]
        self.run = run
        self.waiting = False
        self.reads = 0

    def read(self):
        pos = super().read()
        if self.waiting and self._arrived("present"):
            self.reads += 1
            if self.run is None or 0 < self.reads % (self.run + 2) <= self.run:
                pos["gripper"] = self.value
        return pos


@pytest.mark.parametrize("value", ["closed", "open"])
def test_glitches_shorter_than_take_reads_are_ignored(value):
    """5 reads in a row (a quarter of a second) is noise: neither a take nor a drop."""
    arm = GlitchyClaw(value, 5)
    r = hand(arm, handoff_timeout_s=3, on_state=lambda s: setattr(arm, "waiting", s == "wait_for_take"))
    assert r["result"] == "timeout", r["detail"]
    assert arm.reads >= 40                 # read at about 20 Hz for the 3 s
    assert arm.object == "held" and where(arm) == "lift"


def test_claw_reading_open_while_waiting_is_dropped():
    arm = GlitchyClaw("open", None)
    r = hand(arm, on_state=lambda s: setattr(arm, "waiting", s == "wait_for_take"))
    assert r["result"] == "dropped"
    assert pick.TAKE_READS <= arm.reads <= pick.TAKE_READS + 15    # decided on the 6th read, then let go
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]


@pytest.mark.parametrize("slip_s", [0.0, 0.5, 0.9])   # 0.9: starts closing just inside TAKE_SETTLE_S
def test_slip_at_present_is_dropped_not_taken(slip_s):
    """The person reaches only once the arm has stopped (plan 17): a claw closing within
    TAKE_SETTLE_S of arriving lost the object on its own."""
    arm = SimArm(never_take=True, slip_at_present_after_s=slip_s)
    on_state, seen, _ = watch_states(arm)
    r = hand(arm, on_state=on_state)
    assert r["result"] == "dropped"
    assert r["detail"] == "object slipped out at the present pose before anyone could take it"
    assert [s[0] for s in seen] == ["pick", "present", "wait_for_take"]
    assert arm.clock.now() - seen[-1][3] < pick.TAKE_SETTLE_S + 3.0 + 4.0   # no waiting out the timeout
    assert arm.object == "floor"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]


def test_slip_after_the_settle_window_looks_like_a_take():
    """Known limit (plan 10.7): a slip later than TAKE_SETTLE_S closes the claw just as a take
    does, so it is reported "taken" though nobody took it."""
    arm = SimArm(never_take=True, slip_at_present_after_s=5.0)
    r = hand(arm)
    assert r["result"] == "taken"
    assert arm.object == "floor"


class SlipOnTheWayBackToLift(SimArm):
    """Nobody takes it, and the object falls out as the arm leaves present after the timeout."""

    def __init__(self, **kw):
        super().__init__(never_take=True, **kw)
        self.was_presented = False

    def step(self, dt):
        super().step(dt)
        if self.presented_s > 0:
            self.was_presented = True
        elif self.object == "held" and self.was_presented:
            self.object = "floor"
            self._missing = True


def test_timeout_with_nothing_left_in_the_claw_is_dropped():
    """Plan 9.4: "timeout" means the arm is at lift, still holding."""
    arm = SlipOnTheWayBackToLift()
    r = hand(arm, handoff_timeout_s=3)
    assert r["result"] == "dropped"
    assert "lift" in r["detail"]
    assert arm.object == "floor"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]


@pytest.mark.parametrize("ticks, at", [(3, "present"), (40, None)])
def test_stop_while_letting_go_after_a_take_holds(ticks, at):
    """3 ticks: while the claw opens at present; 40: on the way back to ready."""
    arm = SimArm()
    letting_go = lambda: arm.object == "hand" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]  # noqa: E731
    stop = stop_when(arm, letting_go, ticks)
    r = hand(arm, stop_event=stop)
    assert r["result"] == "stopped"
    assert arm.clock.now() - stop.at <= 0.5
    assert all(arm.goal[j] == arm.pos[j] for j in poses_mod.ARM_JOINTS)
    assert where(arm) == at


def test_stop_on_the_way_back_to_lift_holds_and_keeps_squeezing():
    arm = SimArm(never_take=True)
    waited = {}
    stop = stop_when(arm, lambda: "wait_for_take" in waited and arm.clock.now() >= waited["wait_for_take"] + 3.5)
    r = hand(arm, handoff_timeout_s=3, stop_event=stop,
             on_state=lambda s: waited.setdefault(s, arm.clock.now()))
    assert r["result"] == "stopped"
    assert arm.clock.now() - stop.at <= 0.5
    assert where(arm) is None and arm.object == "held"      # between present and lift
    assert holding_still(arm)


def test_sim_take_timer_starts_once_the_arm_has_stopped_at_present():
    arm = SimArm(take_after_s=0.0)
    arm.object, arm.pos["gripper"] = "held", arm._object_stop      # holding it, squeezing
    nearly = dict(DEFAULT_POSES["present"], elbow_flex=DEFAULT_POSES["present"]["elbow_flex"] + 3)
    arm.pos.update({j: nearly[j] for j in poses_mod.ARM_JOINTS})
    arm.command({**{j: nearly[j] for j in poses_mod.ARM_JOINTS}, "gripper": SQUEEZE_TARGET})
    for _ in range(30):
        arm.clock.sleep(0.05)            # within 5 deg of present, but not there
    assert arm.object == "held" and arm.presented_s == 0
    arm.command({j: DEFAULT_POSES["present"][j] for j in poses_mod.ARM_JOINTS})
    arm.clock.sleep(0.05)
    assert arm.object == "hand"
    with pytest.raises(ValueError, match="slip_at_present_after_s"):
        SimArm(slip_at_present_after_s=-1)


def test_hand_mode_needs_present_before_any_motion():
    poses = {k: v for k, v in DEFAULT_POSES.items() if k != "present"}
    arm = Recorder(poses=poses)
    with pytest.raises(poses_mod.PosesError, match="'present' is missing; hand mode needs it"):
        pick.pick_to_hand(arm, poses)
    assert arm.sent == []
    assert pick.pick_to_tray(SimArm(poses), poses)["result"] == "in_tray"   # tray mode doesn't need it


def test_hand_mode_does_not_need_above_tray():
    poses = {k: v for k, v in DEFAULT_POSES.items() if k != "above_tray"}
    arm = SimArm(poses)
    assert pick.pick_to_hand(arm, poses)["result"] == "taken"
    assert arm.object == "hand"
    tray_arm = Recorder(poses=poses)
    with pytest.raises(poses_mod.PosesError, match="'above_tray' is missing; tray mode needs it"):
        pick.pick_to_tray(tray_arm, poses)
    assert tray_arm.sent == []


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, None, True, "60"])
def test_rejects_bad_handoff_timeout(timeout):
    arm = Recorder()
    with pytest.raises(ValueError, match="handoff_timeout_s"):
        hand(arm, handoff_timeout_s=timeout)
    assert arm.sent == []


# --- A7: the command line in hand mode --------------------------------------------

@pytest.mark.parametrize("extra, result, code", [
    ([], "taken", 0),
    (["--to", "hand"], "taken", 0),
    (["--sim-never-take", "--handoff-timeout", "3"], "timeout", 5),
    (["--sim-miss", "2"], "missed", 2),
    (["--sim-slip"], "dropped", 3),
    (["--sim-stall", "shoulder_lift"], "stuck", 4),
    (["--to", "tray"], "in_tray", 0),
])
def test_cli_hand_mode_dry_run(tmp_path, capsys, extra, result, code):
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", *extra)
    assert (rc, r["result"]) == (code, result), r["detail"]
    assert set(r) == {"result", "grasp", "attempts", "seconds", "detail"}
    if result in ("taken", "timeout"):
        assert "state present" in out and "state wait_for_take" in out


def test_cli_handoff_timeout_with_tray_is_an_argument_error(tmp_path, capsys, fake_real_arm):
    rc, r, _ = run_cli(tmp_path, capsys, "--to", "tray", "--handoff-timeout", "30", "--poses", str(fake_real_arm))
    assert (rc, r["result"]) == (1, "error")
    assert "--handoff-timeout" in r["detail"] and "--to hand" in r["detail"]
    assert NoHardware.opened == []


def test_cli_stop_while_waiting_is_exit_1(tmp_path, capsys, monkeypatch):
    class SignalWhileWaiting(SimArm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sleep = self.clock.sleep

            def sleep_then_signal(dt):
                sleep(dt)
                if self.presented_s > 1.0:
                    signal.raise_signal(signal.SIGINT)

            self.clock.sleep = sleep_then_signal

    monkeypatch.setattr(sim_mod, "SimArm", SignalWhileWaiting)
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", "--sim-never-take")
    assert (rc, r["result"]) == (1, "stopped")
    assert "state wait_for_take" in out


def test_cli_sim_take_after(tmp_path, capsys):
    _, quick, _ = run_cli(tmp_path, capsys, "--dry-run")
    _, slow, _ = run_cli(tmp_path, capsys, "--dry-run", "--sim-take-after", "7")
    assert quick["result"] == slow["result"] == "taken"
    assert slow["seconds"] - quick["seconds"] == pytest.approx(5.0, abs=0.2)


def capture_timeout(monkeypatch):
    seen = []
    real = pick.pick_to_hand

    def spy(*args, **kwargs):
        seen.append(kwargs["handoff_timeout_s"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pick, "pick_to_hand", spy)
    return seen


@pytest.mark.parametrize("extra, expected", [([], 7.0), (["--handoff-timeout", "3"], 3.0)])
def test_cli_handoff_timeout_defaults_to_robot_yaml(tmp_path, capsys, monkeypatch, extra, expected):
    seen = capture_timeout(monkeypatch)
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", "--sim-never-take", *extra,
                       config=write_config(tmp_path, handoff={"timeout_s": 7, "grip": "squeeze"}))
    assert (rc, r["result"]) == (5, "timeout")
    assert seen == [expected]
    assert f"{expected:g} s" in r["detail"]


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_refuses_bad_handoff_timeout_before_opening_the_arm(tmp_path, capsys, monkeypatch, fake_real_arm,
                                                                value, mode):
    seen = capture_timeout(monkeypatch)
    extra = ["--dry-run"] if mode == "dry" else []
    rc, r, _ = run_cli(tmp_path, capsys, "--handoff-timeout", value, "--poses", str(fake_real_arm), *extra)
    assert (rc, r["result"]) == (1, "error")
    assert "--handoff-timeout" in r["detail"]
    assert seen == [] and NoHardware.opened == []


@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_bad_handoff_config_is_a_clear_error_in_hand_mode_only(tmp_path, capsys, fake_real_arm, mode):
    extra = ["--dry-run"] if mode == "dry" else []
    config = write_config(tmp_path, handoff={"timeout_s": 0})
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm), *extra, config=config)
    assert (rc, r["result"]) == (1, "error")
    assert r["detail"].startswith("ConfigError:") and "handoff.timeout_s" in r["detail"]
    assert NoHardware.opened == []
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm), "--to", "tray", *extra, config=config)
    assert (rc, r["result"]) == (0, "in_tray")        # tray mode doesn't use it


def test_cli_hand_mode_refuses_poses_without_present_before_opening_the_arm(tmp_path, capsys, fake_real_arm):
    path = tmp_path / "old_poses.json"
    poses_mod.save({k: v for k, v in DEFAULT_POSES.items() if k != "present"}, path)
    for extra in ([], ["--dry-run"]):
        rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(path), *extra)
        assert (rc, r["result"]) == (1, "error")
        assert "'present' is missing" in r["detail"] and "python -m arm.teach" in r["detail"]
    assert NoHardware.opened == []
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(path), "--dry-run", "--to", "tray")
    assert (rc, r["result"]) == (0, "in_tray")


# --- A7: robot.yaml handoff.timeout_s (no fallback) --------------------------------

@pytest.mark.parametrize("handoff, words", [
    (DROP, "'handoff:' section"),
    ("60", "'handoff:' section"),
    ({}, "handoff.timeout_s"),
    ({"timeout_s": None}, "handoff.timeout_s"),
    ({"timeout_s": 0}, "handoff.timeout_s"),
    ({"timeout_s": -5}, "handoff.timeout_s"),
    ({"timeout_s": "60"}, "handoff.timeout_s"),
    ({"timeout_s": True}, "handoff.timeout_s"),
    ({"timeout_s": math.nan}, "handoff.timeout_s"),
    ({"timeout_s": math.inf}, "handoff.timeout_s"),
])
def test_load_handoff_timeout_rejects_bad_values(tmp_path, handoff, words):
    with pytest.raises(pick.ConfigError, match=words.replace(".", r"\.")):
        pick.load_handoff_timeout(write_config(tmp_path, handoff=handoff))


def test_load_handoff_timeout(tmp_path):
    assert pick.load_handoff_timeout(write_config(tmp_path)) == 60.0
    assert pick.load_handoff_timeout(write_config(tmp_path, handoff={"timeout_s": 12.5})) == 12.5
    with pytest.raises(pick.ConfigError, match="not found"):
        pick.load_handoff_timeout(tmp_path / "none.yaml")


# --- D9: grip while presenting, squeeze or light (plan 10.1, 13) ----------------------

HOLD_POS = SimArm()._object_stop                  # where the sim's claw stops on the object
LIGHT_TARGET = HOLD_POS - pick.LIGHT_SQUEEZE      # DEFAULT_POSES: closing lowers the value


def light(arm, **kwargs):
    return hand(arm, handoff_grip="light", **kwargs)


def test_light_grip_taken_after_a_pull():
    """Pulled out of a light hold, the claw closes only LIGHT_SQUEEZE, short of the closed side;
    TAKE_CLOSE_UNITS closed from where it held the object counts as closed."""
    arm = Recorder(take_after_s=2.0)
    on_state, seen, _ = watch_states(arm)
    logs = []
    r = light(arm, on_state=on_state, log=lambda msg: logs.append((msg, arm.clock.now())))
    assert (r["result"], r["grasp"], r["attempts"]) == ("taken", "scripted", 1), r["detail"]
    assert [s[:3] for s in seen] == [("pick", "floor", "ready"), ("present", "held", "lift"),
                                     ("wait_for_take", "held", "present")]
    assert arm.object == "hand"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]
    taken_at = next(t for msg, t in logs if msg == "taken")
    pulled_at = seen[-1][3] + 2.0
    assert pulled_at < taken_at <= pulled_at + (pick.TAKE_READS + 2) / pick.TAKE_POLL_HZ   # decided promptly
    assert ("wait for take (light grip)", seen[-1][3]) in logs
    grips = [c["gripper"] for c in arm.sent]
    assert set(grips[:seen[-1][4]]) == {DEFAULT_POSES["gripper_open"], SQUEEZE_TARGET}   # carry moves squeeze
    waiting = grips[seen[-1][4]:grips.index(DEFAULT_POSES["gripper_open"], seen[-1][4])]
    assert len(waiting) >= 2 * pick.TAKE_POLL_HZ and set(waiting) == {LIGHT_TARGET}


def test_light_grip_timeout_goes_back_to_lift_squeezing():
    arm = Recorder(never_take=True)
    on_state, seen, _ = watch_states(arm)
    r = light(arm, handoff_timeout_s=5, on_state=on_state)
    assert r["result"] == "timeout", r["detail"]
    assert "5 s" in r["detail"]
    grips = [c["gripper"] for c in arm.sent]
    assert set(grips[:seen[-1][4]]) == {DEFAULT_POSES["gripper_open"], SQUEEZE_TARGET}   # carry moves squeeze
    waiting = grips[seen[-1][4]:]
    back = waiting.index(SQUEEZE_TARGET)               # the move back to lift squeezes again
    assert back >= 5 * pick.TAKE_POLL_HZ and set(waiting[:back]) == {LIGHT_TARGET}
    assert set(waiting[back:]) == {SQUEEZE_TARGET}
    assert where(arm) == "lift"
    assert arm.object == "held" and claw_holding(arm.pos["gripper"], DEFAULT_POSES["gripper_open"],
                                                 DEFAULT_POSES["gripper_closed"])


@pytest.mark.parametrize("slip_s", [0.0, 0.5, 0.9])   # 0.9: starts closing just inside TAKE_SETTLE_S
def test_light_grip_slip_at_present_is_dropped_not_taken(slip_s):
    """The light hold closes only to LIGHT_TARGET after a slip, short of the closed side,
    and that still reads as the object slipping out."""
    arm = SimArm(never_take=True, slip_at_present_after_s=slip_s)
    on_state, seen, _ = watch_states(arm)
    r = light(arm, on_state=on_state)
    assert r["result"] == "dropped"
    assert r["detail"] == "object slipped out at the present pose before anyone could take it"
    assert [s[0] for s in seen] == ["pick", "present", "wait_for_take"]
    assert arm.clock.now() - seen[-1][3] < pick.TAKE_SETTLE_S + 3.0 + 4.0   # no waiting out the timeout
    assert arm.object == "floor"
    assert where(arm) == "ready" and arm.goal["gripper"] == DEFAULT_POSES["gripper_open"]


class ClawStopsShort(SimArm):
    """After the take the claw closes only `short` units from where it held the object
    (on a finger, say), short of the closed side."""

    def __init__(self, short, **kw):
        super().__init__(**kw)
        self.short = short

    def step(self, dt):
        super().step(dt)
        if self.object == "hand":
            self.pos["gripper"] = max(self.pos["gripper"], self._object_stop - self.short)


@pytest.mark.parametrize("grip", pick.HANDOFF_GRIPS)
def test_closing_take_close_units_from_the_hold_is_a_take_only_in_light_grip(grip):
    """Light: closing TAKE_CLOSE_UNITS from the hold is a take. Squeeze keeps main's rule (the
    closed side only), so a soft case creeping a few units under the full squeeze isn't."""
    arm = ClawStopsShort(pick.TAKE_CLOSE_UNITS, never_take=False)
    short_of_closed_side = DEFAULT_POSES["gripper_closed"] + pick.GRIP_MARGIN
    assert arm._object_stop - pick.TAKE_CLOSE_UNITS >= short_of_closed_side
    r = hand(arm, handoff_grip=grip, handoff_timeout_s=5)
    if grip == "light":
        assert r["result"] == "taken", r["detail"]
        assert arm.object == "hand" and where(arm) == "ready"
    else:
        assert r["result"] == "timeout", r["detail"]


@pytest.mark.parametrize("width, light_squeeze", [(15.0, pick.LIGHT_SQUEEZE), (5.0, pick.LIGHT_SQUEEZE),
                                                  (5.0, 12.0)])   # 12 past a 5-wide object is past g_squeeze
@pytest.mark.parametrize("ending", ["taken", "stopped"])
def test_light_wait_claw_command_stays_near_the_hold(monkeypatch, width, light_squeeze, ending):
    """While waiting, the claw is never commanded more than LIGHT_SQUEEZE past where it held the
    object (measured, so it follows the object's width), nor past the full squeeze. STOP keeps
    that target."""
    monkeypatch.setattr(pick, "LIGHT_SQUEEZE", light_squeeze)
    arm = Recorder(object_width=width, never_take=ending == "stopped")
    on_state, seen, stop = watch_states(arm, "wait_for_take", 40)
    r = light(arm, on_state=on_state, stop_event=stop if ending == "stopped" else None)
    assert r["result"] == ending, r["detail"]
    hold_pos = arm._object_stop
    expected = max(hold_pos - light_squeeze, SQUEEZE_TARGET)
    after = [c["gripper"] for c in arm.sent[seen[-1][4]:]]        # the wait, then letting go or STOP
    assert after.count(expected) >= 20
    assert all(hold_pos - g <= light_squeeze + 1e-9 and g >= SQUEEZE_TARGET for g in after)
    if ending == "stopped":
        assert where(arm) == "present" and arm.object == "held"
        assert all(arm.goal[j] == arm.pos[j] for j in poses_mod.ARM_JOINTS)
        assert arm.goal["gripper"] == expected


@pytest.mark.parametrize("sim, expected", [({}, "taken"), ({"never_take": True}, "timeout"),
                                           ({"never_take": True, "slip_at_present_after_s": 0.5}, "dropped")])
def test_squeeze_grip_is_the_default_and_unchanged(sim, expected):
    default, squeeze = Recorder(**sim), Recorder(**sim)
    r = hand(default, handoff_timeout_s=3)
    assert r["result"] == expected
    assert hand(squeeze, handoff_timeout_s=3, handoff_grip="squeeze") == r
    assert squeeze.sent == default.sent
    assert {c["gripper"] for c in squeeze.sent} == {DEFAULT_POSES["gripper_open"], SQUEEZE_TARGET}


@pytest.mark.parametrize("grip", ["firm", "Light", " light", "", None, True, 1])
def test_rejects_bad_handoff_grip(grip):
    arm = Recorder()
    with pytest.raises(ValueError, match="handoff_grip"):
        hand(arm, handoff_grip=grip)
    assert arm.sent == []


@pytest.mark.parametrize("handoff, words", [
    (DROP, "'handoff:' section"),
    ({"timeout_s": 60}, "handoff.grip"),
    ({"timeout_s": 60, "grip": None}, "handoff.grip"),
    ({"timeout_s": 60, "grip": ""}, "handoff.grip"),
    ({"timeout_s": 60, "grip": "firm"}, "handoff.grip"),
    ({"timeout_s": 60, "grip": "Squeeze"}, "handoff.grip"),
    ({"timeout_s": 60, "grip": "light "}, "handoff.grip"),
    ({"timeout_s": 60, "grip": True}, "handoff.grip"),
    ({"timeout_s": 60, "grip": 6}, "handoff.grip"),
    ({"timeout_s": 60, "grip": ["light"]}, "handoff.grip"),
])
def test_load_handoff_grip_rejects_bad_values(tmp_path, handoff, words):
    with pytest.raises(pick.ConfigError, match=words.replace(".", r"\.")):
        pick.load_handoff_grip(write_config(tmp_path, handoff=handoff))


def test_load_handoff_grip(tmp_path):
    assert pick.load_handoff_grip(write_config(tmp_path)) == "squeeze"
    assert pick.load_handoff_grip(write_config(tmp_path, handoff={"timeout_s": 60, "grip": "light"})) == "light"
    with pytest.raises(pick.ConfigError, match="not found"):
        pick.load_handoff_grip(tmp_path / "none.yaml")


def capture_grip(monkeypatch):
    seen = []
    real = pick.pick_to_hand

    def spy(*args, **kwargs):
        seen.append(kwargs["handoff_grip"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pick, "pick_to_hand", spy)
    return seen


@pytest.mark.parametrize("config_grip, extra, expected", [
    ("squeeze", [], "squeeze"),
    ("light", [], "light"),
    ("squeeze", ["--handoff-grip", "light"], "light"),
    ("light", ["--handoff-grip", "squeeze"], "squeeze"),
])
def test_cli_handoff_grip_defaults_to_robot_yaml(tmp_path, capsys, monkeypatch, config_grip, extra, expected):
    seen = capture_grip(monkeypatch)
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", *extra,
                         config=write_config(tmp_path, handoff={"timeout_s": 60, "grip": config_grip}))
    assert (rc, r["result"]) == (0, "taken"), r["detail"]
    assert seen == [expected]
    assert f"wait for take ({expected} grip)" in out


def test_cli_light_grip_timeout(tmp_path, capsys):
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", "--handoff-grip", "light", "--sim-never-take",
                         "--handoff-timeout", "3")
    assert (rc, r["result"]) == (5, "timeout"), r["detail"]
    assert "wait for take (light grip)" in out


@pytest.mark.parametrize("handoff", [{"timeout_s": 60}, {"timeout_s": 60, "grip": "firm"}])
@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_bad_handoff_grip_config_is_a_clear_error_in_hand_mode_only(tmp_path, capsys, monkeypatch,
                                                                        fake_real_arm, handoff, mode):
    seen = capture_grip(monkeypatch)
    extra = ["--dry-run"] if mode == "dry" else []
    config = write_config(tmp_path, handoff=handoff)
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm), *extra, config=config)
    assert (rc, r["result"]) == (1, "error")
    assert r["detail"].startswith("ConfigError:") and "handoff.grip" in r["detail"]
    assert seen == [] and NoHardware.opened == []
    rc, r, _ = run_cli(tmp_path, capsys, "--poses", str(fake_real_arm), "--to", "tray", *extra, config=config)
    assert (rc, r["result"]) == (0, "in_tray")        # tray mode doesn't use it


@pytest.mark.parametrize("extra, words", [(["--handoff-grip", "firm"], "--handoff-grip"),
                                          (["--handoff-grip", "Light"], "--handoff-grip"),
                                          (["--to", "tray", "--handoff-grip", "light"], "--to hand")])
def test_cli_bad_handoff_grip_flag_is_an_argument_error(tmp_path, capsys, monkeypatch, fake_real_arm, extra, words):
    seen = capture_grip(monkeypatch)
    rc, r, _ = run_cli(tmp_path, capsys, *extra, "--poses", str(fake_real_arm))
    assert (rc, r["result"]) == (1, "error")
    assert "--handoff-grip" in r["detail"] and words in r["detail"]
    assert seen == [] and NoHardware.opened == []



# --- frozen wrist roll (servo 5 never called) ---------------------------------------

class FrozenRollArm(SimArm):
    """Like RealArm with arm.frozen_joints [wrist_roll]: the wrist never moves, reads 0."""
    frozen = ("wrist_roll",)

    def read(self):
        return {**super().read(), "wrist_roll": 0.0}

    def command(self, targets):
        super().command({j: v for j, v in targets.items() if j != "wrist_roll"})


def test_the_hand_off_works_with_the_wrist_roll_frozen():
    r = pick.pick_to_hand(FrozenRollArm(), DEFAULT_POSES)
    assert r["result"] == "taken", r["detail"]


@pytest.mark.parametrize("entry", [pick.pick_to_hand, pick.pick_to_tray])
def test_a_roll_offset_is_refused_when_the_wrist_roll_is_frozen(entry):
    with pytest.raises(ValueError, match="wrist_roll is frozen"):
        entry(FrozenRollArm(), DEFAULT_POSES, roll_deg=5)


@pytest.mark.parametrize("value, ok", [(None, True), ([], True), (["wrist_roll"], True),
                                       (["gripper"], False), (["wrist_roll", "wrist_roll"], False),
                                       ("wrist_roll", False), (["elbow"], False)])
def test_frozen_joints_config(tmp_path, value, ok):
    path = write_config(tmp_path, frozen_joints=value)
    if ok:
        assert pick.load_arm_config(path)["frozen_joints"] == (value or [])
    else:
        with pytest.raises(pick.ConfigError, match="frozen_joints"):
            pick.load_arm_config(path)


def test_frozen_joints_default_to_none(tmp_path):
    assert pick.load_arm_config(write_config(tmp_path))["frozen_joints"] == []



@pytest.mark.parametrize("mode", ["dry", "real"])
def test_cli_refuses_a_roll_offset_when_the_wrist_roll_is_frozen(tmp_path, capsys, fake_real_arm, mode):
    extra = ["--dry-run"] if mode == "dry" else []
    rc, r, _ = run_cli(tmp_path, capsys, "--roll", "5", "--poses", str(fake_real_arm), *extra,
                       config=write_config(tmp_path, frozen_joints=["wrist_roll"]))
    assert (rc, r["result"]) == (1, "error") and "frozen" in r["detail"]
    assert NoHardware.opened == []                       # refused before any arm opened
