from pathlib import Path
import yaml
from base.parking import Parking


def controller():
    return Parking(yaml.safe_load((Path(__file__).parents[1] / 'config/robot.yaml').read_text()))


def test_stop_on_missing_stale_or_invalid_floor():
    for floor, age in [(None, 0), ([.5, 0], .4), ([float('nan'), 0], 0)]:
        cmd = controller().update(floor, frame=1, age=age, now=0)
        assert (cmd.left, cmd.right, cmd.done) == (0, 0, False)


def test_settle_requires_distinct_frames():
    p = controller()
    for frame in [1, 1, 1, 2]:
        assert not p.update(p.target, frame=frame, age=0, now=0).done
    assert p.update(p.target, frame=3, age=0, now=0).done


def test_direction_speed_and_overshoot():
    p = controller()
    cmd = p.update([.5, .2], frame=1, age=0, now=0)
    assert cmd.right > cmd.left
    assert max(abs(cmd.left), abs(cmd.right)) <= 30
    cmd = p.update([.1, 0], frame=2, age=0, now=1)
    assert cmd.left == cmd.right == 0 and not cmd.done
    assert 'passed' in cmd.reason
    assert p.update([.5, 0], frame=3, age=0, now=2).left == 0


def test_timeout_stays_stopped():
    p = controller()
    p.update([.5, 0], frame=1, age=0, now=0)
    cmd = p.update([.5, 0], frame=2, age=0, now=121)
    assert cmd.left == cmd.right == 0 and 'timed out' in cmd.reason


def test_ideal_straight_approach_converges():
    p = controller()
    x = .5
    for frame in range(1000):
        cmd = p.update([x, 0], frame=frame, age=0, now=frame * .1)
        x -= (cmd.left + cmd.right) / 2 / 1000 * .1
        if cmd.done:
            break
    assert cmd.done and abs(x - p.target[0]) <= p.tolerance


def test_pickup_range_guard_even_inside_parking_tolerance():
    p = controller()
    p.target = (.32, 0)
    # .34 is within the .03 parking tolerance but beyond the .33 reach limit.
    for frame in range(5):
        command = p.update([.34, 0], frame=frame, age=0, now=0)
        assert not command.done
        assert command.left > 0 and command.right > 0
    for frame in range(5, 8):
        command = p.update([.325, 0], frame=frame, age=0, now=0)
    assert command.done


def test_pickup_range_rejects_unreachable_parking_configuration():
    import pytest
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'config/robot.yaml').read_text())
    cfg['perception']['parking_spot_floor_m'] = [.5, 0]
    with pytest.raises(ValueError, match='exceeds pickup range'):
        Parking(cfg)
