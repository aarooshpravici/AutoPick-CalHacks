import copy
from pathlib import Path
import pytest
import yaml
from base.driver import BaseDriver, FakeSerial

ROOT = Path(__file__).parents[1]


def config():
    return yaml.safe_load((ROOT / 'config/robot.yaml').read_text())['base']


def test_disabled_outputs_reported_and_refused_for_real_motion():
    cfg = config()
    with BaseDriver(FakeSerial(config=cfg), max_speed_mm_s=30, max_lift_mm=1000) as driver:
        report = driver.verify_firmware(cfg, require_ready=False)
        assert not report['wiring'] and not report['estop_sensed']
        with pytest.raises(RuntimeError, match='outputs disabled'):
            driver.verify_firmware(cfg)


@pytest.mark.parametrize('line', [
    'ERR invalid command',
    'INFO other-board 2 limit=1 hold=1 watchdog_ms=300 wiring=1 estop_sense=0 max_mm_s=30 config=0000000000000000',
    'INFO floorpicker-r4 1 limit=1 hold=1 watchdog_ms=300 wiring=1 estop_sense=0 max_mm_s=30 config=0000000000000000',
    'INFO floorpicker-r4 2 limit=0 hold=1 watchdog_ms=300 wiring=1 estop_sense=0 max_mm_s=30 config=0000000000000000',
    'INFO floorpicker-r4 2 limit=1 hold=1 watchdog_ms=999 wiring=1 estop_sense=0 max_mm_s=30 config=0000000000000000',
    'INFO floorpicker-r4 2 limit=1 hold=1 watchdog_ms=300 wiring=1 estop_sense=0 max_mm_s=nan config=0000000000000000',
])
def test_incompatible_firmware_never_drives(line):
    port = FakeSerial()
    original = port.write
    def write(data):
        if data == b'INFO\n':
            with port._lock:
                port._lines.append((line+'\n').encode())
            return len(data)
        return original(data)
    port.write = write
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        with pytest.raises(RuntimeError):
            driver.verify_firmware(config())
        assert not any(c.startswith('DRIVE') for c in port.commands)


def test_original_firmware_error_is_visible_to_user():
    port = FakeSerial()
    with BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000) as driver:
        with pytest.raises(RuntimeError, match='ERR INFO unavailable'):
            driver.firmware_info()


def test_partial_write_fault_cannot_be_renewed():
    port = FakeSerial()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    original = port.write
    try:
        driver.status(timeout=1)
        port.write = lambda data: len(data)-1
        with pytest.raises(OSError, match='partial serial write'):
            driver.drive(5, 5)
        with pytest.raises(RuntimeError, match='partial serial write'):
            driver.status()
    finally:
        port.write = original
        driver.close()


def test_disconnect_fault_stays_visible_and_prevents_new_drive():
    import time
    class Disconnected(FakeSerial):
        unplugged = False
        def readline(self):
            if self.unplugged:
                raise OSError('USB unplugged')
            return super().readline()
    port = Disconnected()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000)
    try:
        driver.status(timeout=1)
        port.unplugged = True
        deadline = time.monotonic()+1
        while driver._fault is None and time.monotonic()<deadline:
            time.sleep(.005)
        with pytest.raises(RuntimeError, match='USB unplugged'):
            driver.drive(5, 5)
        assert not any(c.startswith('DRIVE ') for c in port.commands)
    finally:
        driver.close()
