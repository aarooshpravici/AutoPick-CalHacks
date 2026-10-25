"""Review fixes: no writes to unidentified devices, firmware flag cross-checks, faithful FakeSerial."""
from pathlib import Path
from unittest.mock import Mock
import time
import pytest
import yaml
from base.driver import BaseDriver, FakeSerial
from base.jog import jog
from base.link import main as link_main

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / 'config/robot.yaml').read_text())


class Silent:
    def __init__(self):
        self.writes = []

    def readline(self):
        time.sleep(.005)
        return b''

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def close(self):
        pass


def test_close_writes_nothing_to_silent_unidentified_port():
    port = Silent()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    with pytest.raises(TimeoutError):
        driver.status(timeout=.1)
    driver.close()
    assert port.writes == []


def test_close_still_stops_an_identified_device():
    port = FakeSerial()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    driver.status(timeout=1)
    driver.close()
    assert port.commands == ['STOP']


@pytest.mark.parametrize('key', ['follower_port', 'leader_port'])
@pytest.mark.parametrize('swap', [False, True])
def test_link_refuses_arm_ports_before_opening(config, tmp_path, monkeypatch, key, swap):
    import serial
    opened = Mock(side_effect=AssertionError('opened hardware'))
    monkeypatch.setattr(serial, 'Serial', opened)
    path = tmp_path / 'robot.yaml'
    path.write_text(yaml.safe_dump(config))
    port = config['arm'][key]
    if swap:
        port = port.replace('/dev/tty.', '/dev/cu.')
    with pytest.raises(SystemExit):
        link_main(['--config', str(path), '--port', port])
    opened.assert_not_called()


def tamper(port, old, new):
    original = port.write

    def write(data):
        result = original(data)
        if data == b'INFO\n':
            with port._lock:
                port._lines.append(port._lines.pop().decode().replace(old, new).encode())
        return result
    port.write = write


@pytest.mark.parametrize('old,new', [('wiring=0', 'wiring=1'), ('estop_sense=0', 'estop_sense=1')])
def test_info_flags_disagreeing_with_yaml_are_refused_even_when_hash_matches(config, old, new):
    base = config['base']
    port = FakeSerial(config=base)
    tamper(port, old, new)
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        with pytest.raises(RuntimeError, match='differ from robot.yaml'):
            driver.verify_firmware(base, require_ready=False)


def test_info_flags_agreeing_with_yaml_pass(config):
    base = dict(config['base'], wiring_confirmed=True, estop_sense_pin=7)
    with BaseDriver(FakeSerial(config=base), max_speed_mm_s=30, max_lift_mm=1000) as driver:
        info = driver.verify_firmware(base)
        assert info['wiring'] and info['estop_sensed']


def test_jog_real_flag_is_explicit(config):
    port = FakeSerial(config=config['base'])
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        with pytest.raises(RuntimeError, match='outputs disabled'):
            jog(driver, config, 'forward', 5, .05, 10, real=True)
        assert not any(c.startswith('DRIVE') for c in port.commands)


def fake_driver(**options):
    port = FakeSerial(**options)
    driver = BaseDriver(port, max_speed_mm_s=200, max_lift_mm=1000)
    driver.status(timeout=1)
    return port, driver


def test_fake_refuses_drive_without_limit():
    port, driver = fake_driver()
    with driver:
        driver.drive(10, 10)
        with pytest.raises(RuntimeError, match='travel limit required'):
            driver.ping()
    assert port.left == port.right == 0


def test_fake_zero_drive_needs_no_limit():
    port, driver = fake_driver()
    with driver:
        driver.drive(0, 0)
        driver.ping()


def test_fake_with_wiring_false_refuses_drive_and_reports_it(config):
    port, driver = fake_driver(config=dict(config['base'], wiring_confirmed=True), wiring=False)
    with driver:
        assert not driver.firmware_info()['wiring']
        driver.set_travel_limit(100)
        driver.drive(10, 10)
        with pytest.raises(RuntimeError, match='unconfigured motors'):
            driver.ping()
    assert port.left == port.right == 0


def test_fake_repeats_limit_reached_with_status():
    port = FakeSerial()
    port.write(b'LIMIT 1\n')
    port.write(b'DRIVE 100 100\n')
    time.sleep(.05)
    seen = []
    deadline = time.monotonic() + .6
    while time.monotonic() < deadline and seen.count(b'LIMIT REACHED\n') < 3:
        seen.append(port.readline())
    assert seen.count(b'LIMIT REACHED\n') >= 3


def test_stale_stop_ack_cannot_satisfy_new_stop():
    port = FakeSerial()
    original = port.write
    port._lines.append(b'OK STOP\n')  # Left over from an earlier session.

    def write(data):
        if data == b'STOP\n':
            port.commands.append('STOP')
            return len(data)  # Firmware never acknowledges this one.
        return original(data)
    port.write = write
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        driver.status(timeout=1)
        with pytest.raises(TimeoutError, match='STOP acknowledgment'):
            driver.synchronize(timeout=.3)


def test_stale_stop_ack_is_drained_then_real_ack_accepted():
    port = FakeSerial()
    port._lines.append(b'OK STOP\n')
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        driver.status(timeout=1)
        driver.synchronize()
        assert driver._stop_acked == driver._stop_generation == 1


def test_synchronize_absorbs_ack_for_stop_left_pending_by_earlier_session():
    port = FakeSerial()
    original = port.write

    def write(data):
        result = original(data)
        if data == b'STOP\n':  # The earlier session's pending STOP is acknowledged alongside ours.
            with port._lock:
                port._lines.append(b'OK STOP\n')
        return result
    port.write = write
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        driver.status(timeout=1)
        driver.synchronize()
        assert driver._stop_acked == driver._stop_generation
        port.write = original
        driver.stop_and_wait()


def test_extra_stop_ack_is_a_link_fault():
    port = FakeSerial()
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        driver.status(timeout=1)
        driver.synchronize()
        with port._lock:
            port._lines.append(b'OK STOP\n')
            port._lines.append(b'OK STOP\n')
        with pytest.raises(RuntimeError, match='extra STOP'):
            driver.stop_and_wait()


def test_original_exception_survives_stop_failure_in_close():
    port = FakeSerial()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    driver.status(timeout=1)
    driver.stop = Mock(side_effect=OSError('stop failed'))
    with pytest.raises(ValueError, match='body'):
        with driver:
            raise ValueError('body')
    assert port.closed


def test_close_failure_still_raises_without_body_exception():
    port = FakeSerial()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    driver.status(timeout=1)
    driver.stop = Mock(side_effect=OSError('stop failed'))
    with pytest.raises(OSError, match='stop failed'):
        with driver:
            pass
    assert port.closed
