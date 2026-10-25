import copy
import json
from pathlib import Path
import socket
import threading
import time

import pytest
import yaml

from arm.sim import SimArm
from base.driver import BaseDriver, FakeSerial
from brain.detections import FakeDetections, ReplayDetections, UdpDetections
from brain.mission import Mission, build


def wait_for(check, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.005)
    raise AssertionError("timed out")


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/robot.yaml').read_text())


@pytest.fixture
def make(config, tmp_path):
    brains = []
    def create(**options):
        port = options.pop('port', FakeSerial())
        driver = BaseDriver(port, max_speed_mm_s=200, max_lift_mm=1000)
        driver.status(timeout=1)
        brain = Mission(config, base=driver, arm=options.pop('arm', SimArm()),
                        source=options.pop('source', FakeDetections()), period=.01,
                        log_dir=tmp_path, **options)
        brains.append(brain)
        return brain
    yield create
    for brain in brains:
        brain.close()


def test_full_simulated_mission_uses_real_pick_and_logs_states(make):
    brain = make()
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['last_pick'])
    assert brain.status()['state'] == 'Idle'
    assert brain.status()['last_pick']['result'] == 'taken'
    assert brain.arm.object == 'hand'
    brain.close()
    records = [json.loads(line) for line in brain.log_path.read_text().splitlines()]
    states = [row['state'] for row in records if row['event'] == 'state']
    assert states[:6] == ['Search', 'Approach', 'Pick', 'Present', 'WaitForTake', 'Idle']
    assert any(row['event'] == 'detection' for row in records)
    assert any(row.get('command') == 'STOP' for row in records)
    assert brain.status()['lift_mm'] is None
    assert brain.status()['tray_g'] is None


@pytest.mark.parametrize('options,result', [
    ({'never_take': True}, 'timeout'), ({'miss_first': 2}, 'missed'),
    ({'stall_joint': 'elbow_flex'}, 'stuck'), ({'slip_on_lift': True}, 'dropped')])

def test_real_simarm_failure_paths_and_attempt_limit(make, options, result):
    brain = make(arm=SimArm(**options))
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'Failed')
    assert brain.status()['last_pick']['result'] == result
    if result == 'dropped':
        assert brain._attempts == 3
    else:
        assert brain._attempts == 1


def test_stop_during_handoff_is_fast_and_latched(make):
    waiting = threading.Event()
    def pick(arm, poses, *, stop_event, on_state, **kwargs):
        on_state('present'); on_state('wait_for_take'); waiting.set()
        assert stop_event.wait(2)
        return dict(result='stopped')
    brain = make(pick=pick)
    brain.request('phone', 'seated')
    assert waiting.wait(2)
    start = time.monotonic()
    brain.stop()
    assert time.monotonic() - start < .05
    assert brain.status()['state'] == 'Stopped'
    wait_for(lambda: not brain._pick_thread.is_alive())
    time.sleep(.05)
    assert brain.status()['state'] == 'Stopped'
    assert brain.base.port.commands[-1] == 'STOP'


def test_stop_never_waits_for_serial_io(make):
    brain = make()
    entered, release = threading.Event(), threading.Event()
    original = brain.base.port.write
    def slow_write(data):
        if data.startswith(b'DRIVE'):
            entered.set(); release.wait(1)
        return original(data)
    brain.base.port.write = slow_write
    brain.request('phone', 'seated')
    assert entered.wait(1)
    start = time.monotonic()
    brain.stop()
    assert time.monotonic() - start < .05
    assert brain.status()['state'] == 'Stopped'
    release.set()
    wait_for(lambda: brain.base.port.commands[-1] == 'STOP')
    assert brain.base.port.left == brain.base.port.right == 0


def test_replacement_waits_for_old_arm_and_ignores_old_result(make):
    started, held, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def pick(arm, poses, *, stop_event, **kwargs):
        calls.append(stop_event)
        if len(calls) == 1:
            started.set(); assert stop_event.wait(2)
            held.set(); assert release.wait(2)
            return dict(result='taken')  # stale result must not finish replacement
        return dict(result='taken')
    brain = make(pick=pick)
    brain.request('phone', 'seated')
    assert started.wait(1)
    brain.request('phone', 'standing')
    assert held.wait(1)
    time.sleep(.1)
    assert len(calls) == 1
    assert brain.status()['state'] == 'Search'
    release.set()
    wait_for(lambda: len(calls) == 2 and brain.status()['state'] == 'Idle')


def test_camera_loss_stops_search(make):
    class Lost:
        def poll(self): return None
        def close(self): pass
    brain = make(source=Lost())
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'Failed', timeout=2)
    assert brain.status()['message'] == 'Camera lost'
    wait_for(lambda: brain.base.port.commands[-1] == 'STOP')


def test_empty_floor_search_exhaustion(make):
    class Empty(FakeDetections):
        def poll(self):
            message = super().poll(); message['items'] = []; return message
    brain = make(source=Empty())
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'Failed')
    assert 'Two simulated turns' in brain.status()['message']
    assert brain.status()['last_pick'] is None


def test_build_real_refuses_without_opening_hardware(config):
    with pytest.raises(RuntimeError, match='P4'):
        build(config, sim=False)


def test_udp_and_replay_messages(config, make, tmp_path):
    source = UdpDetections('127.0.0.1:0')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(b'{bad', source.socket.getsockname())
        stale = FakeDetections().poll(); stale['t'] -= 5
        sender.sendto(json.dumps(stale).encode(), source.socket.getsockname())
        message = FakeDetections().poll()
        sender.sendto(json.dumps(message).encode(), source.socket.getsockname())
        received = wait_for(source.poll)
    assert received == message and source.rejected == 2
    source.close()
    path = tmp_path / 'replay.jsonl'
    path.write_text('\n'.join(json.dumps(dict(message=FakeDetections().poll())) for _ in range(30)))
    brain = make(source=ReplayDetections(path))
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['last_pick'])
    assert brain.status()['last_pick']['result'] == 'taken'


def test_ui_routes_use_real_mission_stop_and_status(make):
    from fastapi.testclient import TestClient
    from ui.server import create_app
    brain = make()
    with TestClient(create_app(brain, parse=lambda text: dict(action='fetch', item='phone'))) as client:
        assert client.post('/request', json={'text': 'phone'}).status_code == 200
        response = client.post('/stop')
        assert response.status_code == 200
        assert response.json()['brain_ms'] < 50
        assert client.get('/status').json()['state'] == 'Stopped'
        assert client.get('/thumb/wrist').status_code == 404
