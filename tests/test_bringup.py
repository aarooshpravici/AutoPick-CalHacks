"""No-hardware tests for human-run bounded bring-up and read-only preflight."""
import copy
from pathlib import Path
from unittest.mock import Mock
import pytest
import yaml
from base.driver import BaseDriver, FakeSerial
from base.jog import jog, validate, main
from brain.check import inspect

ROOT = Path(__file__).parents[1]

@pytest.fixture
def config():
    return yaml.safe_load((ROOT / 'config/robot.yaml').read_text())

@pytest.mark.parametrize('direction,signs', [('forward',(1,1)), ('back',(-1,-1)), ('left',(-1,1)), ('right',(1,-1))])
def test_jog_directions_are_budgeted_then_stopped(config, direction, signs):
    port = FakeSerial()
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        result = jog(driver, config, direction, 5, .08, 10)
        assert result['stop_acknowledged']
        assert port.commands[:2] == ['STOP', 'LIMIT 10.000000']
        assert f'DRIVE {5*signs[0]} {5*signs[1]}' in port.commands
        assert port.commands[-1] == 'STOP'
        assert port.left == port.right == 0

@pytest.mark.parametrize('speed,duration,distance', [(11,1,10),(5,3,10),(5,1,21),(float('nan'),1,10),(5,1,0)])
def test_jog_rejects_unsafe_limits(config, speed, duration, distance):
    with pytest.raises(ValueError):
        validate(config, 'forward', speed, duration, distance)


def test_jog_attempts_stop_on_interruption(config):
    driver = Mock()
    driver.limit_reached = False
    driver.drive.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        jog(driver, config, 'forward', 5, 1, 10)
    assert driver.stop_and_wait.call_count == 2


def test_real_jog_refuses_noninteractive_before_opening_ports(config, tmp_path, monkeypatch):
    config['base']['wiring_confirmed'] = True
    path = tmp_path / 'robot.yaml'; path.write_text(yaml.safe_dump(config))
    import serial
    opened = Mock(side_effect=AssertionError('opened hardware'))
    monkeypatch.setattr(serial, 'Serial', opened)
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    with pytest.raises(SystemExit):
        main(['--config', str(path), '--real', '--port', '/dev/cu.R4', '--direction', 'forward'])
    opened.assert_not_called()


def test_preflight_reports_missing_artifacts_and_never_opens_ports(config, tmp_path, monkeypatch):
    import serial
    opened = Mock(side_effect=AssertionError('opened hardware'))
    monkeypatch.setattr(serial, 'Serial', opened)
    path = tmp_path / 'robot.yaml'; path.write_text(yaml.safe_dump(config))
    checks = {row['check']:row for row in inspect(path, 'robot', root=tmp_path)}
    for key in ('base configuration','R4 port','grasp and poses','floor and pickup target','phone checkpoint'):
        assert not checks[key]['ok']
    assert checks['arm configuration and frozen wrist']['ok']
    opened.assert_not_called()
