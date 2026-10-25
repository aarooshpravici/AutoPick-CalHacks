"""Dry-run tests for the suction grasp (pick, sim, poses, command line). No hardware.

The suction cup is passive (an Octobuddy): no pump, no sensor. The arm presses it on the phone,
dwells, lifts, presents, holds a fixed time for the pull, and goes back to ready, without ever
moving the claw. Teaching and the preflight check have their cases in test_teach.py and
test_check.py.
"""

import json

import pytest
import yaml

import arm.real as real_mod
import arm.sim as sim_mod
import threading
from arm import pick, poses as poses_mod
from arm.poses import ARM_JOINTS, JOINTS, SUCTION_POSES
from arm.sim import DEFAULT_POSES, SimArm

CLAW = DEFAULT_POSES["suction_press"]["gripper"]
SUCTION_ONLY = {k: v for k, v in DEFAULT_POSES.items() if k == "ready" or k in SUCTION_POSES}
MOVES = ["move ready", "move suction_above", "move suction_press", "dwell 2 s", "move suction_above",
         "move suction_lift", "move suction_present", "hold for the pull 4 s", "move ready"]


class Recorder(SimArm):
    """A sim that keeps every command it was sent."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent = []

    def command(self, targets):
        self.sent.append(dict(targets))
        super().command(targets)


def suction(arm, poses=DEFAULT_POSES, **kwargs):
    return pick.pick_to_hand(arm, poses, grasp="suction", **kwargs)


def logged(arm):
    """(log function, [(message, clock, commands sent so far)])."""
    seen = []
    return (lambda msg: seen.append((msg, arm.clock.now(), len(getattr(arm, "sent", []))))), seen


def at(arm, name, tolerance=1.0):
    return all(abs(arm.pos[j] - DEFAULT_POSES[name][j]) <= tolerance for j in ARM_JOINTS)


def steps(sent):
    """The biggest arm-joint move between each pair of consecutive commands."""
    return [max(abs(b[j] - a[j]) for j in ARM_JOINTS) for a, b in zip(sent, sent[1:])]


def stop_during(arm, message, after_s):
    """(stop_event, log): the event is set once the clock is after_s past the log line `message`."""
    stop = threading.Event()
    marks = {}
    sleep = arm.clock.sleep

    def sleep_then_stop(dt):
        sleep(dt)
        if message in marks and arm.clock.now() >= marks[message] + after_s:
            stop.set()

    arm.clock.sleep = sleep_then_stop
    return stop, lambda msg: marks.setdefault(msg, arm.clock.now())


# --- the sequence --------------------------------------------------------------

def test_clean_suction_pick_ends_taken_back_at_ready():
    arm = Recorder()
    log, seen = logged(arm)
    r = suction(arm, log=log)
    assert (r["result"], r["grasp"], r["attempts"]) == ("taken", "suction", 1), r["detail"]
    assert set(r) == {"result", "grasp", "attempts", "seconds", "detail"}
    assert arm.object == "hand" and arm.grasps == 1       # it stuck, and the person pulled it off
    assert at(arm, "ready")
    assert "no sensor" in r["detail"]                      # the result says nothing was checked
    assert [m for m, _, _ in seen] == MOVES


def test_states_are_reported_like_the_claw_hand_off():
    states = []
    suction(SimArm(), on_state=states.append)
    assert states == ["pick", "present", "wait_for_take"]


def test_the_claw_never_moves():
    arm = Recorder()
    suction(arm)
    assert arm.sent and all(c["gripper"] == CLAW for c in arm.sent)
    assert arm.goal["gripper"] == CLAW


def test_the_claw_target_is_the_one_taught_in_suction_press():
    poses = {**DEFAULT_POSES, "suction_press": {**DEFAULT_POSES["suction_press"], "gripper": 44.0}}
    arm = Recorder()
    suction(arm, poses)
    assert {c["gripper"] for c in arm.sent} == {44.0}


@pytest.mark.parametrize("dwell, hold", [(2.0, 4.0), (1.5, 0.5)])
def test_dwell_and_hold_last_as_long_as_asked(dwell, hold):
    arm = Recorder()
    log, seen = logged(arm)
    args = {} if (dwell, hold) == (2.0, 4.0) else {"suction_dwell_s": dwell, "suction_hold_s": hold}
    suction(arm, log=log, **args)
    t = {}
    for message, now, _ in seen:
        t.setdefault(message, []).append(now)
    after_dwell = t["move suction_above"][1] - t[f"dwell {dwell:g} s"][0]
    held = t["move ready"][-1] - t[f"hold for the pull {hold:g} s"][0]
    assert after_dwell == pytest.approx(dwell, abs=0.1)
    assert held == pytest.approx(hold, abs=0.1)


def test_the_defaults_are_a_2_s_dwell_and_a_4_s_timed_hold():
    assert (pick.SUCTION_DWELL_S, pick.SUCTION_HOLD_S) == (2.0, 4.0)


def test_the_press_and_the_lift_off_are_gentle_and_the_present_is_slow():
    arm = Recorder()
    log, seen = logged(arm)
    suction(arm, speed_deg_s=45, log=log)
    mark = {}
    for index, (message, _, n) in enumerate(seen):
        mark.setdefault(message, []).append((index, n))
    n_press = mark["move suction_press"][0][1]
    n_dwell = mark["dwell 2 s"][0][1]
    n_off = mark["move suction_above"][1][1]
    n_lift = mark["move suction_lift"][0][1]
    n_present = mark["move suction_present"][0][1]
    gentle = pick.SUCTION_PRESS_SPEED_DEG_S / pick.RATE_HZ
    # Slices start at a move's first command: the step from the previous move's last command is the
    # arm's one-tick lag behind it, not a speed.
    assert max(steps(arm.sent[n_press:n_dwell])) == pytest.approx(gentle)          # onto the phone
    assert max(steps(arm.sent[n_off:n_lift])) == pytest.approx(gentle)             # straight back up
    assert max(steps(arm.sent[n_lift:n_present])) == pytest.approx(45 / pick.RATE_HZ)   # then full speed
    near_person = steps(arm.sent[n_present:])                                      # present, hold, ready
    assert max(near_person) == pytest.approx(pick.PRESENT_SPEED_DEG_S / pick.RATE_HZ)


@pytest.mark.parametrize("speed", [10, 25])
def test_a_slow_run_is_never_sped_up_by_the_gentle_or_present_speeds(speed):
    arm = Recorder()
    suction(arm, speed_deg_s=speed)
    assert max(steps(arm.sent)) == pytest.approx(speed / pick.RATE_HZ)


def test_pan_and_roll_offsets_reach_the_poses_that_touch_the_phone_and_no_others():
    # Distinct pan and roll per pose, so an offset on the wrong move can't hide behind equal values.
    poses = json.loads(json.dumps(DEFAULT_POSES))
    for name, (pan, roll) in {"ready": (0.0, 0.0), "suction_above": (1.0, 2.0), "suction_press": (3.0, 4.0),
                              "suction_lift": (5.0, 6.0), "suction_present": (8.0, 9.0)}.items():
        poses[name]["shoulder_pan"], poses[name]["wrist_roll"] = pan, roll
    arm = Recorder(poses=poses)
    log, seen = logged(arm)
    suction(arm, poses, pan_deg=7.0, roll_deg=-3.0, log=log)

    def sent_during(first, last):       # commands from the first `first` line to the last `last` line
        n0 = next(n for m, _, n in seen if m == first)
        n1 = [n for m, _, n in seen if m == last][-1]
        return arm.sent[n0 + 1:n1]

    def reached(cmds, pan, roll):
        return any(abs(c["shoulder_pan"] - pan) < 1e-6 and abs(c["wrist_roll"] - roll) < 1e-6 for c in cmds)

    assert reached(sent_during("move suction_above", "move suction_press"), 1.0 + 7.0, 2.0 - 3.0)
    assert reached(sent_during("move suction_press", "dwell 2 s"), 3.0 + 7.0, 4.0 - 3.0)
    held = sent_during("hold for the pull 4 s", "move ready")
    assert held and all(abs(c["shoulder_pan"] - 8.0) < 1e-6 and abs(c["wrist_roll"] - 9.0) < 1e-6
                        for c in held)      # the present pose carries no offset
    assert arm.sent[-1]["shoulder_pan"] == 0.0 and arm.sent[-1]["wrist_roll"] == 0.0


# --- what the sim's passive cup does -------------------------------------------

def test_a_press_shorter_than_the_seal_time_does_not_stick():
    arm = SimArm(seal_after_s=3.0)
    r = suction(arm)                          # 2 s dwell, plus the few tenths of a second arriving and leaving
    assert arm.object == "floor" and arm.grasps == 0
    assert r["result"] == "taken"             # known limit: nothing can sense it, so the run still ends taken


def test_a_longer_dwell_sticks_where_a_shorter_one_did_not():
    arm = SimArm(seal_after_s=3.0)
    suction(arm, suction_dwell_s=3.5)
    assert arm.object == "hand"


def test_the_first_presses_that_miss_leave_the_phone_behind_then_it_sticks():
    arm = SimArm(miss_first=1)
    suction(arm)
    assert arm.object == "floor" and arm.grasps == 0
    suction(arm)
    assert arm.object == "hand" and arm.grasps == 1


def test_a_phone_that_falls_off_while_lifting_is_not_noticed():
    arm = SimArm(slip_on_lift=True)
    r = suction(arm)
    assert arm.object == "floor"
    assert r["result"] == "taken"             # known limit: no sensor, same as a miss


def test_a_phone_nobody_pulls_comes_back_to_ready_still_on_the_cup():
    arm = SimArm(never_take=True)
    r = suction(arm)
    assert arm.object == "suction" and at(arm, "ready")
    assert r["result"] == "taken"             # known limit: the timed hold cannot tell


def test_a_pull_after_the_hold_has_ended_is_too_late():
    arm = SimArm(take_after_s=5.0)
    suction(arm)
    assert arm.object == "suction"


def test_the_person_pulls_inside_the_hold():
    arm = SimArm(take_after_s=3.5)
    suction(arm)
    assert arm.object == "hand"


# --- stopping and failing -------------------------------------------------------

def test_stop_during_the_dwell_holds_the_arm_pressed_where_it_is():
    arm = SimArm()
    stop, log = stop_during(arm, "dwell 2 s", 1.2)     # long enough for the cup to have sealed
    r = suction(arm, stop_event=stop, log=log)
    assert r["result"] == "stopped"
    assert all(arm.goal[j] == arm.pos[j] for j in ARM_JOINTS) and arm.goal["gripper"] == CLAW
    assert at(arm, "suction_press") and arm.object == "suction"


def test_stop_during_the_hold_at_present_holds_the_arm_there():
    arm = SimArm(never_take=True)
    stop, log = stop_during(arm, "hold for the pull 4 s", 1.0)
    r = suction(arm, stop_event=stop, log=log)
    assert r["result"] == "stopped"
    assert all(arm.goal[j] == arm.pos[j] for j in ARM_JOINTS) and arm.goal["gripper"] == CLAW
    assert at(arm, "suction_present") and arm.object == "suction"


def test_stop_on_the_way_to_present_holds_where_it_is():
    arm = SimArm()
    stop, log = stop_during(arm, "move suction_present", 0.2)
    r = suction(arm, stop_event=stop, log=log)
    assert r["result"] == "stopped"
    assert all(arm.goal[j] == arm.pos[j] for j in ARM_JOINTS)
    assert not at(arm, "suction_present") and not at(arm, "ready")


def test_a_stuck_joint_ends_stuck_holding():
    arm = SimArm(stall_joint="elbow_flex")
    r = suction(arm)
    assert r["result"] == "stuck" and "elbow_flex" in r["detail"]
    assert arm.goal["elbow_flex"] == arm.pos["elbow_flex"]
    assert arm.object == "floor"              # it never got as far as pressing


# --- refused before any motion ---------------------------------------------------

def test_missing_suction_poses_are_refused_with_the_teach_hint():
    claw_only = {k: v for k, v in DEFAULT_POSES.items() if k not in SUCTION_POSES}
    arm = Recorder()
    with pytest.raises(poses_mod.PosesError, match=r"suction_above.*python -m arm\.teach --grasp suction"):
        suction(arm, claw_only)
    assert arm.sent == []


def test_a_missing_ready_is_refused():
    poses = {k: v for k, v in SUCTION_ONLY.items() if k != "ready"}
    arm = Recorder()
    with pytest.raises(poses_mod.PosesError, match="ready"):
        suction(arm, poses)
    assert arm.sent == []


@pytest.mark.parametrize("kwargs", [
    {"grasp": "vacuum"}, {"grasp": None},
    {"policy": "some/checkpoint"},
    {"suction_dwell_s": 0}, {"suction_dwell_s": -1}, {"suction_dwell_s": float("nan")},
    {"suction_dwell_s": "2"}, {"suction_dwell_s": True},
    {"suction_hold_s": 0}, {"suction_hold_s": float("inf")}, {"suction_hold_s": None},
    {"speed_deg_s": 90}, {"speed_deg_s": 0},
    {"pan_deg": float("nan")},
])
def test_bad_arguments_are_refused_before_the_arm_moves(kwargs):
    arm = Recorder()
    merged = {"grasp": "suction", **kwargs}
    with pytest.raises(ValueError):
        pick.pick_to_hand(arm, DEFAULT_POSES, **merged)
    assert arm.sent == []


def test_a_frozen_wrist_roll_refuses_a_roll_offset():
    arm = Recorder()
    arm.frozen = ("wrist_roll",)
    with pytest.raises(ValueError, match="frozen"):
        suction(arm, roll_deg=5.0)
    assert arm.sent == []


def test_camera_size_is_refused_without_a_policy():
    with pytest.raises(ValueError, match="camera_size"):
        suction(SimArm(), camera_size=(640, 360))


def test_the_claw_hand_off_is_unchanged_and_ignores_the_suction_poses():
    arm = SimArm()
    r = pick.pick_to_hand(arm, DEFAULT_POSES)
    assert (r["result"], r["grasp"]) == ("taken", "scripted")
    assert arm.grasps == 1 and arm.object == "hand"


# --- poses: suction mode needs no claw poses or values ---------------------------

def test_a_suction_only_pose_set_runs():
    arm = SimArm(SUCTION_ONLY)
    r = suction(arm, SUCTION_ONLY)
    assert (r["result"], arm.object) == ("taken", "hand")


def test_suction_only_poses_validate_for_suction_only():
    poses_mod.validate(SUCTION_ONLY, "suction")
    for mode in ("hand", "tray", None):
        with pytest.raises(poses_mod.PosesError):
            poses_mod.validate(SUCTION_ONLY, mode)


def test_claw_values_that_are_there_are_still_checked_in_suction_mode():
    with pytest.raises(poses_mod.PosesError, match="gripper_closed"):
        poses_mod.validate({**SUCTION_ONLY, "gripper_open": 60.0}, "suction")
    with pytest.raises(poses_mod.PosesError, match="less than"):
        poses_mod.validate({**SUCTION_ONLY, "gripper_open": 60.0, "gripper_closed": 59.0}, "suction")


def test_every_suction_pose_must_have_all_six_joints():
    broken = {**SUCTION_ONLY, "suction_lift": {j: 0.0 for j in JOINTS if j != "elbow_flex"}}
    with pytest.raises(poses_mod.PosesError, match="suction_lift.*elbow_flex"):
        poses_mod.validate(broken, "suction")


def test_suction_poses_round_trip_beside_the_claw_poses(tmp_path):
    path = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, path)
    again = poses_mod.load(path)
    for name in SUCTION_POSES:
        assert again[name] == {j: float(DEFAULT_POSES[name][j]) for j in JOINTS}
    assert [n for n in poses_mod.POSE_NAMES if n in again][-4:] == SUCTION_POSES


def test_a_suction_only_file_saves_without_claw_values_and_loads_in_suction_mode(tmp_path):
    path = tmp_path / "poses.json"
    poses_mod.save(SUCTION_ONLY, path, "suction")
    on_disk = json.loads(path.read_text())
    assert "gripper_open" not in on_disk and "gripper_closed" not in on_disk
    assert poses_mod.load(path, "suction")["suction_press"]["gripper"] == CLAW
    with pytest.raises(poses_mod.PosesError):
        poses_mod.load(path)


def test_saving_a_suction_only_file_without_the_mode_is_refused(tmp_path):
    with pytest.raises(poses_mod.PosesError):
        poses_mod.save(SUCTION_ONLY, tmp_path / "poses.json")
    assert not (tmp_path / "poses.json").exists()


def test_load_partial(tmp_path):
    assert poses_mod.load_partial(tmp_path / "missing.json") == {}
    path = tmp_path / "poses.json"
    poses_mod.save(DEFAULT_POSES, path)
    partial = poses_mod.load_partial(path)
    assert "units" not in partial and set(partial) >= {"ready", "gripper_open", *SUCTION_POSES}
    path.write_text("{not json")
    with pytest.raises(poses_mod.PosesError, match="not valid JSON"):
        poses_mod.load_partial(path)
    path.write_text("[1, 2]")
    with pytest.raises(poses_mod.PosesError, match="JSON object"):
        poses_mod.load_partial(path)


def test_the_teach_hint_for_each_mode_names_a_real_command():
    assert poses_mod.TEACH_HINT["suction"] == "python -m arm.teach --grasp suction"
    assert set(poses_mod.MODE_POSES) == {"hand", "tray", "suction"}


# --- the sim itself ----------------------------------------------------------------

def test_sim_rejects_a_bad_seal_time():
    for bad in (-1, "1", None, True):
        with pytest.raises(ValueError, match="seal_after_s"):
            SimArm(seal_after_s=bad)


def test_the_claw_does_not_decide_whether_the_cup_sticks():
    """A claw standing wide open, or squeezed shut, makes no difference to the cup."""
    for claw in (2.0, 60.0):
        poses = {**DEFAULT_POSES, "suction_press": {**DEFAULT_POSES["suction_press"], "gripper": claw}}
        arm = SimArm(poses)
        suction(arm, poses)
        assert (arm.object, arm.grasps) == ("hand", 1), claw


# --- command line ---------------------------------------------------------------------

ROBOT_YAML = {
    "arm": {"follower_port": "/dev/tty.usbmodemTEST1", "leader_port": "/dev/tty.usbmodemTEST2",
            "follower_id": "so101_follower_arm", "leader_id": "so101_leader_arm",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
    "handoff": {"timeout_s": 60, "grip": "squeeze"},
}


def write_config(tmp_path, with_handoff=True):
    data = json.loads(json.dumps(ROBOT_YAML))
    if not with_handoff:
        del data["handoff"]
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def run_cli(tmp_path, capsys, *extra, poses=None, config=None):
    argv = ["run", "--poses", str(poses or tmp_path / "none.json"), "--config",
            str(config or write_config(tmp_path)), *extra]
    rc = pick.main(argv)
    out = capsys.readouterr().out
    last = out.strip().splitlines()[-1]
    assert last.startswith("RESULT ")
    return rc, json.loads(last[len("RESULT "):]), out


def test_cli_dry_run_suction(tmp_path, capsys):
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction")
    assert (rc, r["result"], r["grasp"]) == (0, "taken", "suction"), r["detail"]
    assert "state present" in out and "state wait_for_take" in out and "dwell 2 s" in out


def test_cli_dwell_and_hold_flags(tmp_path, capsys):
    _, default, _ = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction")
    rc, short, out = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction",
                             "--suction-dwell", "1", "--suction-hold", "1")
    assert rc == 0 and "dwell 1 s" in out and "hold for the pull 1 s" in out
    assert default["seconds"] - short["seconds"] == pytest.approx(4.0, abs=0.2)


@pytest.mark.parametrize("extra, result, code", [
    (["--sim-never-take"], "taken", 0),         # known limit: a hold nobody pulled in still ends taken
    (["--sim-slip"], "taken", 0),               # ... and so does a phone that fell off
    (["--sim-miss", "1"], "taken", 0),          # ... and a cup that did not stick
    (["--sim-stall", "shoulder_lift"], "stuck", 4),
])
def test_cli_dry_run_outcomes(tmp_path, capsys, extra, result, code):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction", *extra)
    assert (rc, r["result"]) == (code, result), r["detail"]


@pytest.mark.parametrize("extra, words", [
    (["--to", "tray"], ["--grasp suction", "--to hand"]),
    (["--policy", "some/checkpoint"], ["--policy", "--grasp suction"]),
    (["--handoff-timeout", "30"], ["--handoff-timeout", "--grasp claw"]),
    (["--handoff-grip", "light"], ["--handoff-grip", "--grasp claw"]),
    (["--suction-dwell", "0"], ["--suction-dwell", "positive"]),
    (["--suction-hold", "-3"], ["--suction-hold", "positive"]),
    (["--suction-hold", "nan"], ["--suction-hold", "positive"]),
    (["--pan", "nan"], ["--pan"]),
])
def test_cli_bad_suction_arguments_end_in_a_result_and_exit_1(tmp_path, capsys, extra, words):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction", *extra)
    assert (rc, r["result"]) == (1, "error")
    assert all(w in r["detail"] for w in words), r["detail"]


@pytest.mark.parametrize("extra", [["--suction-dwell", "2"], ["--suction-hold", "4"]])
def test_cli_suction_timings_need_the_suction_grasp(tmp_path, capsys, extra):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", *extra)
    assert (rc, r["result"], r["grasp"]) == (1, "error", "scripted")
    assert "--grasp suction" in r["detail"]


@pytest.mark.parametrize("argv", [["--grasp", "suction", "--bogus"], ["--grasp=suction", "--bogus"]])
def test_cli_a_bad_command_line_still_reports_the_suction_grasp(tmp_path, capsys, argv):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", *argv)
    assert (rc, r["result"], r["grasp"]) == (1, "error", "suction")


def test_cli_a_poses_file_with_no_suction_poses_stops_before_anything_opens(tmp_path, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr(real_mod, "RealArm", lambda **kw: opened.append(kw))
    claw_only = tmp_path / "poses.json"
    poses_mod.save({k: v for k, v in DEFAULT_POSES.items() if k not in SUCTION_POSES}, claw_only)
    rc, r, _ = run_cli(tmp_path, capsys, "--grasp", "suction", poses=claw_only)
    assert (rc, r["result"], r["grasp"]) == (1, "error", "suction")
    assert "suction_above" in r["detail"] and "python -m arm.teach --grasp suction" in r["detail"]
    assert opened == []


def test_cli_real_mode_runs_a_suction_only_poses_file_and_needs_no_handoff_settings(tmp_path, capsys,
                                                                                     monkeypatch):
    opened = []

    class FakeRealArm(SimArm):
        def __init__(self, **kwargs):
            opened.append(kwargs)
            super().__init__(SUCTION_ONLY)

    monkeypatch.setattr(real_mod, "RealArm", FakeRealArm)
    path = tmp_path / "poses.json"
    poses_mod.save(SUCTION_ONLY, path, "suction")
    rc, r, _ = run_cli(tmp_path, capsys, "--grasp", "suction", "--speed", "25", poses=path,
                       config=write_config(tmp_path, with_handoff=False))
    assert (rc, r["result"], r["grasp"]) == (0, "taken", "suction"), r["detail"]
    assert opened and opened[0]["port"] == "/dev/tty.usbmodemTEST1"


def test_cli_stop_during_the_hold_is_exit_1(tmp_path, capsys, monkeypatch):
    import signal

    class SignalWhileHolding(SimArm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sleep = self.clock.sleep

            def sleep_then_signal(dt):
                sleep(dt)
                if self.presented_s > 1.0:
                    signal.raise_signal(signal.SIGINT)

            self.clock.sleep = sleep_then_signal

    monkeypatch.setattr(sim_mod, "SimArm", SignalWhileHolding)
    rc, r, out = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "suction", "--sim-never-take")
    assert (rc, r["result"]) == (1, "stopped")
    assert "state wait_for_take" in out


def test_cli_claw_runs_are_unchanged_by_the_new_flag(tmp_path, capsys):
    rc, r, _ = run_cli(tmp_path, capsys, "--dry-run", "--grasp", "claw")
    assert (rc, r["result"], r["grasp"]) == (0, "taken", "scripted")
