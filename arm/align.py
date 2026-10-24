"""Line the leader up with the follower before any teleop starts (SW1's rule).

A live table shows every joint on both arms and the difference. The follower holds
still (its torque is on at its present position, see arm/real.py) while the person
moves the leader. Enter starts teleop only once every joint matches: on an arm
joint within ALIGN_DEG, on the claw within ALIGN_GRIPPER. Until then Enter is
ignored, so the follower never jumps towards a leader that's somewhere else.
Ctrl-C quits with nothing moved.
"""

from __future__ import annotations

import select
import sys
import time

from .poses import JOINTS

ALIGN_DEG = 10.0            # an arm joint matches within this many degrees
ALIGN_GRIPPER = 10.0        # the claw matches within this many units of its 0-100 scale
REFRESH_HZ = 10


class AlignError(RuntimeError):
    pass


def mismatched(leader: dict, follower: dict) -> list[str]:
    """The joints that don't match yet."""
    out = []
    for j in JOINTS:
        tol = ALIGN_GRIPPER if j == "gripper" else ALIGN_DEG
        if abs(leader[j] - follower[j]) > tol:
            out.append(j)
    return out


def table(leader: dict, follower: dict) -> list[str]:
    bad = set(mismatched(leader, follower))
    lines = [f"{'joint':14}{'leader':>9}{'follower':>10}{'diff':>8}"]
    for j in JOINTS:
        unit = "" if j == "gripper" else "°"
        mark = "✗ move the leader" if j in bad else "✓"
        lines.append(f"{j:14}{leader[j]:9.1f}{follower[j]:10.1f}{leader[j] - follower[j]:+7.1f}{unit}  {mark}")
    if bad:
        lines.append(f"Line the leader up with the follower: {len(bad)} joint(s) off. "
                     f"Enter is ignored until every joint shows ✓. Ctrl-C quits.")
    else:
        lines.append("All joints match: press Enter to start.")
    return lines


def _enter_from_stdin(timeout_s: float) -> bool:
    """True if a line arrived on stdin within timeout_s. Raises AlignError on end of input, or
    if stdin can't be watched (not a terminal or pipe): nothing has moved at that point."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (AttributeError, OSError, ValueError) as e:      # io.UnsupportedOperation is an OSError
        raise AlignError(f"can't read Enter from this input ({e}); run it in a terminal. Nothing moved")
    if not ready:
        return False
    if sys.stdin.readline() == "":
        raise AlignError("input closed before Enter was pressed; nothing moved")
    return True


def live_align(leader, follower, *, enter=None, out=None, max_s: float | None = None) -> None:
    """Show the live table until Enter is pressed with every joint matching. Never commands
    the follower. enter(timeout_s) -> bool reports an Enter (default: stdin). max_s, if given,
    gives up with AlignError after that long (for tests and scripts)."""
    enter = enter or _enter_from_stdin
    out = out or sys.stdout
    period = 1.0 / REFRESH_HZ
    drawn = 0
    rejected = False
    t0 = time.monotonic()
    while True:
        lead, foll = leader.read(), follower.read()
        lines = table(lead, foll)
        if rejected:
            lines.append("Not yet: some joints don't match.")
        if drawn:
            out.write(f"\x1b[{drawn}F\x1b[J")      # back to the top of the table, clear below
        out.write("\n".join(lines) + "\n")
        out.flush()
        drawn = len(lines)
        if enter(period):
            if not mismatched(leader.read(), follower.read()):
                out.write("Starting.\n")
                out.flush()
                return
            rejected = True
        if max_s is not None and time.monotonic() - t0 > max_s:
            raise AlignError(f"the joints didn't match within {max_s:g} s; nothing moved")


def scripted_align(leader, follower) -> None:
    """Dry runs: the scripted person puts the simulated leader exactly where the follower is."""
    leader.pos = dict(follower.read())
