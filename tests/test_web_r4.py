"""HTTP -> mission -> pyserial -> PTY -> actual R4 main.cpp; no hardware."""
import os
from pathlib import Path
import pty
import shutil
import subprocess
import threading
import time

import pytest
import serial
import yaml
from fastapi.testclient import TestClient

from arm.sim import SimArm
from base.driver import BaseDriver
from base.link import check
from brain.detections import FakeDetections
from brain.mission import Mission
from brain.navigation import RealNavigation
from ui.server import create_app

ROOT = Path(__file__).parents[1]


def wait_for(check, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('timed out')


@pytest.fixture(scope='module')
def emulator(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('native C++ compiler required for firmware transport integration')
    directory = tmp_path_factory.mktemp('r4-emulator')
    binary = directory / 'controller'
    from base.firmware.generate_config import render
    config = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())['base']
    config['wiring_confirmed'] = True
    (directory / 'pinmap.h').write_text(render(config))
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'tests/arduino'), '-I', str(directory),
                    '-I', str(ROOT / 'base/firmware/include'),
                    str(ROOT / 'tests/r4_serial_emulator.cpp'), '-o', str(binary)], check=True)
    return binary


@pytest.fixture
def wire(emulator):
    master, slave = pty.openpty()
    port = serial.Serial(os.ttyname(slave), baudrate=115200, timeout=.05, write_timeout=.1)
    process = subprocess.Popen([str(emulator)], stdin=master, stdout=master, stderr=subprocess.PIPE, text=True)
    os.close(master); os.close(slave)
    commands = []
    reader = threading.Thread(target=lambda: commands.extend(line.strip() for line in process.stderr), daemon=True)
    reader.start()
    driver = BaseDriver(port, max_speed_mm_s=30, max_lift_mm=1000, timeout=1)
    try:
        driver.status(timeout=2)
        yield driver, commands
    finally:
        driver.close()
        process.terminate(); process.wait(timeout=2)
        reader.join(1); process.stderr.close()


class Source(FakeDetections):
    parked = False
    def poll(self):
        message = super().poll()
        if not self.parked:
            message['items'] = []
        else:
            message['items'][0]['floor_m'] = [.22, 0]
        return message


class MeasuredMapStandIn:
    def map(self, pixel):
        return [.22, 0]


@pytest.mark.parametrize('endpoint,payload', [
    ('/fetch', {'item': 'phone', 'hand': 'standing'}),
    ('/request', {'text': 'I dropped my phone', 'hand': 'seated'})])
def test_http_request_reaches_firmware_and_stops_before_pick(wire, tmp_path, endpoint, payload):
    driver, commands = wire
    config = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())
    source = Source()
    picked = []
    def pick(*args, **kwargs):
        assert commands[-1] == 'STOP'
        assert driver._stop_acked >= driver._stop_generation
        picked.append(True)
        return dict(result='taken')
    brain = Mission(config, base=driver, arm=SimArm(), source=source,
                    navigator_factory=lambda: RealNavigation(config, MeasuredMapStandIn()),
                    pick=pick, log_dir=tmp_path, period=.05)
    def parse(text):
        assert endpoint == '/request', 'Phone click must bypass the language parser'
        return dict(action='fetch', item='phone')
    try:
        with TestClient(create_app(brain, parse=parse)) as client:
            result = client.post(endpoint, json=payload)
            assert result.status_code == 200 and result.json()['parsed']['action'] == 'fetch'
            wait_for(lambda: sum(c.startswith('DRIVE ') for c in commands) >= 3)
            drives = [tuple(map(float, c.split()[1:])) for c in commands if c.startswith('DRIVE ')]
            assert all(max(map(abs, pair)) <= 30 for pair in drives)
            assert any(left < 0 < right for left, right in drives)
            source.parked = True
            wait_for(lambda: picked and brain.status()['last_pick'])
            assert brain.status()['last_pick']['result'] == 'taken'
            source.parked = False
            client.post('/fetch', json={'item': 'phone'})
            wait_for(lambda: brain.status()['state'] == 'Search')
            stop = client.post('/stop')
            assert stop.status_code == 200 and stop.json()['brain_ms'] < 50
            wait_for(lambda: commands[-1] == 'STOP')
            count = len(commands)
            time.sleep(.25)
            assert not any(c.startswith('DRIVE ') for c in commands[count:])
    finally:
        brain.close()


def test_link_probe_never_drives(wire):
    driver, commands = wire
    result = check(driver)
    assert result['pong'] and result['stop_acknowledged']
    assert commands == ['\x18', 'STOP', 'PING', 'INFO']
    assert result['firmware']['limit']
    assert not result['firmware']['estop_sensed']
    assert result['warnings']


def test_selected_item_rejects_serial_injection():
    class Brain:
        calls = []
        def request(self, *args): self.calls.append(args)
        def stop(self): pass
    brain = Brain()
    with TestClient(create_app(brain)) as client:
        for payload in [{'item': 'DRIVE 30 30'}, {'item': 'sock'}, {'item': 'phone', 'hand': 'ceiling'}]:
            assert client.post('/fetch', json=payload).status_code == 422
    assert brain.calls == []

@pytest.mark.parametrize('value', [0, -1, 21, '3', True, 'nan', None])
def test_bad_distance_never_starts(value):
    # Legacy omission remains supported; explicit null is invalid.
    from ui.server import FakeBrain
    with TestClient(create_app(FakeBrain())) as client:
        response = client.post('/fetch', json={'item': 'phone', 'distance_ft': value})
        assert response.status_code == 422


@pytest.mark.parametrize("seen_at_limit", [False, True])
def test_feet_limit_reaches_r4_and_stays_stopped(wire, tmp_path, seen_at_limit):
    driver, commands = wire
    config = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())
    class BoundarySource(Source):
        def poll(self):
            self.parked = seen_at_limit and driver.limit_reached
            return super().poll()
    source = BoundarySource()
    picked = []
    brain = Mission(config, base=driver, arm=SimArm(), source=source,
                    navigator_factory=lambda: RealNavigation(config, MeasuredMapStandIn()),
                    pick=lambda *a, **kw: picked.append(True), log_dir=tmp_path, period=.02)
    try:
        with TestClient(create_app(brain)) as client:
            assert client.post('/fetch', json={'item': 'phone', 'distance_ft': .01}).status_code == 200
            wait_for(lambda: brain.status()['state'] == 'Stopped')
            expected = 'Item seen, but travel limit reached' if seen_at_limit else 'not found at travel limit'
            assert expected in brain.status()['message']
            assert 'LIMIT 3.048000' in commands
            first_drive = next(i for i, c in enumerate(commands) if c.startswith('DRIVE '))
            assert commands.index('LIMIT 3.048000') < first_drive
            stopped_at = time.monotonic()
            wait_for(lambda: driver.status().received_at > stopped_at)
            status = driver.status()
            travel = (abs(status.left_steps) + abs(status.right_steps)) / 2 * (3.14159265 * 71 / 4096)
            assert 0 < travel <= 3.048
            assert not picked
            count = len(commands)
            time.sleep(.3)
            assert not any(c.startswith('DRIVE ') for c in commands[count:])
            # STOP holds the budget latch; a direct accidental DRIVE cannot resume.
            driver.drive(15, 15)
            time.sleep(.15)
            assert driver.limit_reached
            assert driver.status().left_steps == status.left_steps
    finally:
        brain.close()


def test_partial_monitor_drive_is_discarded_without_executing(wire):
    driver, commands = wire
    driver.set_travel_limit(100)
    driver.port.write(b'DRIVE 30 30')  # Leftover line with no newline: must NEVER be executed.
    result = check(driver)
    assert result['stop_acknowledged']
    time.sleep(.15)
    assert driver.status().left_steps == driver.status().right_steps == 0
    assert not any(command == 'DRIVE 30 30' for command in commands)


def test_actual_main_watchdog_stops_even_while_info_traffic_continues(wire):
    driver, _ = wire
    driver.set_travel_limit(100)
    driver.drive(20, 20)
    start = time.monotonic()
    while time.monotonic() - start < .5:
        driver.firmware_info()
        time.sleep(.05)
    stopped = driver.status()
    assert stopped.left_steps > 0
    time.sleep(.2)
    assert driver.status().left_steps == stopped.left_steps
    assert driver.status().right_steps == stopped.right_steps


def test_installed_firmware_configuration_mismatch_refuses_jog(wire):
    from base.jog import jog
    driver, commands = wire
    cfg = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())
    cfg['base']['wiring_confirmed'] = True
    cfg['base']['invert_left'] = True  # Header in this emulator was compiled with false.
    with pytest.raises(RuntimeError, match='configuration differs'):
        jog(driver, cfg, 'forward', 5, .1, 10)
    assert not any(c.startswith('DRIVE ') for c in commands)
    assert commands[-1] == 'STOP'
