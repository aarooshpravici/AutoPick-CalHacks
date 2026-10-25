"""A15 retrace: pure kinematics, no hardware."""
import math

import pytest

from base.homing import Path, Retrace

STEPS_PER_MM = 4096 / (math.pi * 71)
TRACK = 430


def simulate(commands, start=(0.0, 0.0, 0.0, 0.0, 0.0), dt=.05):
    """Integrate (left mm/s, right mm/s) commands; returns x, y, heading, left_steps, right_steps."""
    x, y, th, ls, rs = start
    for left, right in commands:
        v, w = (left + right) / 2, (right - left) / TRACK
        x += v * math.cos(th) * dt; y += v * math.sin(th) * dt; th += w * dt
        ls += left * dt * STEPS_PER_MM; rs += right * dt * STEPS_PER_MM
    return x, y, th, ls, rs


def drive_out():
    """Turn left in place, then straight, then a gentle right arc: the kind of path search+approach makes."""
    plan = [(-15, 15)] * 60 + [(25, 25)] * 200 + [(30, 20)] * 120
    path, state = Path(), (0.0, 0.0, 0.0, 0.0, 0.0)
    path.add(0, 0)
    for command in plan:
        state = simulate([command], state)
        path.add(int(state[3]), int(state[4]))
    return path, state


def test_retrace_returns_near_start_and_original_heading():
    path, state = drive_out()
    assert math.hypot(state[0], state[1]) > 250              # it really went somewhere
    retrace = Retrace(path.points, (int(state[3]), int(state[4])), steps_per_mm=STEPS_PER_MM, speed_mm_s=30)
    for _ in range(20000):
        command = retrace.update(int(state[3]), int(state[4]))
        if command.done:
            break
        assert max(abs(command.left), abs(command.right)) <= 30 + 1e-9
        state = simulate([(command.left, command.right)], state)
    else:
        raise AssertionError("never finished")
    assert math.hypot(state[0], state[1]) < 15                 # mm, commanded steps, no slip
    assert abs(math.degrees(state[2])) < 3


def test_path_spacing_and_length():
    path = Path(spacing=100)
    for i in range(0, 1001, 10):
        path.add(i, i)
    assert len(path.points) == 11
    assert path.length_mm(1) == pytest.approx(1000)


def test_empty_path_refused():
    with pytest.raises(ValueError):
        Retrace([], (0, 0), steps_per_mm=STEPS_PER_MM, speed_mm_s=30)


class LongApproach:
    """Turns, then drives ~0.15 m before parking, so the return has a real path to undo."""

    def __init__(self):
        self.ticks = 0

    def search(self):
        return False, (0, 0)

    def approach(self, detection):
        self.ticks += 1
        return self.ticks > 600, (-30, 30) if self.ticks <= 100 else (30, 30)


def test_mission_drives_back_to_its_starting_steps_before_presenting(tmp_path):
    from pathlib import Path as P
    import time
    import yaml
    from arm.sim import SimArm
    from base.driver import BaseDriver, FakeSerial
    from brain.detections import FakeDetections
    from brain.mission import Mission

    config = yaml.safe_load((P(__file__).resolve().parents[1] / 'config/robot.yaml').read_text())
    base = config['base']
    driver = BaseDriver(FakeSerial(max_speed_mm_s=30, wheel_diameter_mm=base['wheel_diameter_mm'],
                                   steps_per_rev=base['half_steps_per_rev']), max_speed_mm_s=30, max_lift_mm=1000)
    driver.status(timeout=1)
    start = driver.status(timeout=1)
    brain = Mission(config, base=driver, arm=SimArm(), source=FakeDetections(), period=.01,
                    log_dir=tmp_path, navigator_factory=LongApproach)
    try:
        brain.request('phone', 'seated')
        deadline = time.monotonic() + 60
        while not brain.status()['last_pick'] and time.monotonic() < deadline:
            time.sleep(.05)
        assert brain.status()['last_pick']['result'] == 'taken', brain.status()
        end = driver.status(timeout=1)
        assert abs(end.left_steps - start.left_steps) < 120 and abs(end.right_steps - start.right_steps) < 120
        records = [__import__('json').loads(l) for l in brain.log_path.read_text().splitlines()]
        states = [r['state'] for r in records if r['event'] == 'state']
        assert states.index('Return') < states.index('Present')
        ret = next(r for r in records if r['event'] == 'return')
        assert ret["length_mm"] > 60
    finally:
        brain.close()
