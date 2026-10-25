from dataclasses import replace
from pathlib import Path
import yaml
from base.driver import Status
from base.search import Search


def controller():
    return Search(yaml.safe_load((Path(__file__).parents[1] / 'config/robot.yaml').read_text()))


def test_two_turn_budget_and_pause():
    search = controller()
    left = right = 0
    paused = 0
    for tick in range(5000):
        now = tick * .1
        cmd = search.update(Status(round(left), round(right), 0, 0, False, now), now=now)
        if cmd.done:
            break
        assert not cmd.reason
        if cmd.left == cmd.right == 0:
            paused += 1
        left += cmd.left * .1 / search.mm_per_step
        right += cmd.right * .1 / search.mm_per_step
    assert cmd.done and paused >= 24
    assert search.total < 4 * 3.141593 + .1


def test_stale_estop_stall_reverse_stop():
    status = Status(0, 0, 0, 0, False, 0)
    for bad in [replace(status, received_at=-1), replace(status, estop=True)]:
        cmd = controller().update(bad, now=0)
        assert cmd.left == cmd.right == 0 and cmd.reason
    search = controller()
    search.update(status, now=0)
    now = search.timeout + 1
    cmd = search.update(replace(status, received_at=now), now=now)
    assert 'stalled' in cmd.reason and cmd.left == cmd.right == 0
    search = controller()
    search.update(status, now=0)
    cmd = search.update(replace(status, left_steps=10000, received_at=.1), now=.1)
    assert 'reversed' in cmd.reason and cmd.left == cmd.right == 0
