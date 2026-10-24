"""arm/align.py: the live leader/follower table that gates every teleop start. No hardware."""

import io

import pytest

from arm import align
from arm import teach as teach_mod
from arm.poses import JOINTS
from arm.sim import DEFAULT_POSES, SimArm, SimLeader

READY = {**DEFAULT_POSES["ready"], "gripper": DEFAULT_POSES["gripper_open"]}


class Still:
    """An arm that reports a pose and records every command (the follower must get none)."""

    def __init__(self, pose):
        self.pos = dict(pose)
        self.commands = []

    def read(self):
        return dict(self.pos)

    def command(self, targets):
        self.commands.append(dict(targets))


def presses(*script):
    """enter(timeout) that replays a script: True = Enter, False = no key, callables run first."""
    steps = list(script)

    def enter(timeout_s):
        step = steps.pop(0) if steps else False
        if callable(step):
            return step()
        return step
    return enter


def test_enter_is_ignored_until_every_joint_matches():
    follower = Still(READY)
    leader = Still({**READY, "elbow_flex": READY["elbow_flex"] + 30})
    out = io.StringIO()

    def line_up():
        leader.pos = dict(follower.pos)
        return True

    align.live_align(leader, follower, enter=presses(True, False, line_up), out=out, max_s=5)
    text = out.getvalue()
    assert "Not yet" in text and "✗ move the leader" in text
    assert text.rstrip().endswith("Starting.")
    assert follower.commands == []                  # the follower never moves while lining up


@pytest.mark.parametrize("joint, off, matches", [
    ("shoulder_lift", align.ALIGN_DEG - 0.5, True), ("shoulder_lift", align.ALIGN_DEG + 0.5, False),
    ("gripper", align.ALIGN_GRIPPER - 0.5, True), ("gripper", align.ALIGN_GRIPPER + 0.5, False)])
def test_match_tolerances(joint, off, matches):
    leader = {**READY, joint: READY[joint] + off}
    assert (align.mismatched(leader, READY) == []) is matches


def test_table_shows_every_joint_and_the_difference():
    lines = align.table({**READY, "wrist_roll": READY["wrist_roll"] + 25}, READY)
    assert all(any(line.startswith(j) for line in lines) for j in JOINTS)
    roll = next(line for line in lines if line.startswith("wrist_roll"))
    assert "+25.0" in roll and "✗" in roll
    assert "Enter is ignored" in lines[-1]


def test_end_of_input_stops_without_moving():
    follower = Still(READY)

    def closed(timeout_s):
        raise align.AlignError("input closed before Enter was pressed; nothing moved")

    with pytest.raises(align.AlignError, match="nothing moved"):
        align.live_align(Still(READY), follower, enter=closed, out=io.StringIO())
    assert follower.commands == []


def test_gives_up_after_max_s_if_never_matched():
    with pytest.raises(align.AlignError, match="didn't match"):
        align.live_align(Still({**READY, "shoulder_pan": 90}), Still(READY),
                         enter=presses(), out=io.StringIO(), max_s=0.2)


def test_teach_lines_up_before_the_follower_is_ever_commanded():
    follower, leader = SimArm(), SimLeader()
    order = []
    real_command = follower.command
    follower.command = lambda t: (order.append("command"), real_command(t))

    def aligner(lead, foll):
        order.append("align")
        align.scripted_align(lead, foll)

    waiter = teach_mod.scripted_waiter(leader, follower, teach_mod.dry_run_targets(DEFAULT_POSES))
    teach_mod.teach(leader, follower, 45, wait_for=waiter, log=lambda m: None, align=aligner)
    assert order[0] == "align" and order.count("align") == 1


def test_teach_cli_real_mode_uses_the_live_table(monkeypatch, tmp_path):
    import arm.real as real_mod
    used = []
    monkeypatch.setattr(real_mod, "RealLeader", lambda **kw: SimLeader())
    monkeypatch.setattr(real_mod, "RealArm", lambda **kw: SimArm())

    def fake_live(lead, foll):
        used.append("live")
        raise align.AlignError("stopped in the test")

    monkeypatch.setattr(align, "live_align", fake_live)

    def no_keyboard(*args):                         # if alignment were skipped, fail fast, don't hang
        raise teach_mod.TeachError("teaching reached the per-pose Enter without lining up first")

    monkeypatch.setattr(teach_mod, "enter_waiter", no_keyboard)
    cfg = tmp_path / "robot.yaml"
    cfg.write_text("arm:\n  follower_port: /dev/null\n  leader_port: /dev/null\n"
                   "  follower_id: f\n  leader_id: l\n  max_relative_target_deg: 10\n  move_speed_deg_s: 45\n")
    assert teach_mod.main(["--config", str(cfg), "--out", str(tmp_path / "p.json")]) == 1
    assert used == ["live"]


def test_stdin_that_cant_be_watched_fails_safe(monkeypatch):
    class Pseudo:
        def fileno(self):
            import io
            raise io.UnsupportedOperation("redirected stdin is pseudofile, has no fileno()")

    monkeypatch.setattr(align.sys, "stdin", Pseudo())
    follower = Still(READY)
    with pytest.raises(align.AlignError, match="Nothing moved"):
        align.live_align(Still(READY), follower, out=io.StringIO())
    assert follower.commands == []


class TTY(io.StringIO):
    def isatty(self):
        return True


def test_terminal_redraws_in_place_on_the_alternate_screen():
    out = TTY()
    align.live_align(Still(READY), Still(READY), enter=presses(False, False, False, True), out=out, max_s=5)
    text = out.getvalue()
    assert text.startswith(align.ALT_SCREEN_ON)
    assert text.count(align.HOME) == 4                      # one frame per refresh, drawn from the top
    screen, after = text.split(align.ALT_SCREEN_OFF)
    assert "joint" in after and after.rstrip().endswith("Starting.")   # last table left on the normal screen


def test_ctrl_c_still_restores_the_normal_screen():
    out = TTY()

    def ctrl_c(timeout_s):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        align.live_align(Still(READY), Still(READY), enter=ctrl_c, out=out)
    assert align.ALT_SCREEN_OFF in out.getvalue()


def test_not_a_terminal_prints_only_when_the_table_changes():
    out = io.StringIO()
    align.live_align(Still(READY), Still(READY), enter=presses(*[False] * 20, True), out=out, max_s=5)
    assert out.getvalue().count("joint ") == 1                # 21 refreshes, one unchanged table
