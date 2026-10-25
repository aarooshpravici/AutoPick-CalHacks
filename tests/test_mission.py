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


def test_missing_stop_ack_prevents_pick(make):
    class NoAck(FakeSerial):
        def write(self, data):
            result = super().write(data)
            with self._lock:
                self._lines = type(self._lines)(line for line in self._lines if line != b'OK STOP\n')
            return result
    called = []
    brain = make(port=NoAck(), pick=lambda *a, **k: called.append(True))
    brain.request('phone', 'seated')
    wait_for(lambda: brain._io_error)
    assert not called
    assert brain.status()['state'] == 'Failed'
    # Teardown can report the intentionally failed serial link.
    brain._io_error = None


def test_real_gate_does_not_connect_with_unmeasured_base(config, monkeypatch):
    import arm.real
    monkeypatch.setattr(arm.real, 'RealArm', lambda *a, **k: pytest.fail('opened arm'))
    config['base']['wiring_confirmed'] = True
    with pytest.raises(RuntimeError, match='loaded-drive'):
        build(config, sim=False)


def test_old_firmware_rejects_budget_before_any_drive(make):
    class OldFirmware(FakeSerial):
        def write(self, data):
            if data.startswith(b'LIMIT '):
                with self._lock:
                    self.commands.append(data.decode().strip())
                    self._lines.append(b'ERR invalid command\n')
                return len(data)
            return super().write(data)
    port = OldFirmware()
    brain = make(port=port)
    brain.request('phone', 'standing', max_distance_ft=3.5)
    wait_for(lambda: brain.status()['state'] == 'Failed')
    assert 'LIMIT 1066.800000' in port.commands
    assert not any(c.startswith('DRIVE ') for c in port.commands)
    with pytest.raises(RuntimeError, match='STOP FAILED'):
        brain.close()  # Fault remains visible; resources still close in finally.


def test_boundary_camera_timeout_stays_stopped(make):
    port = FakeSerial()
    class FrozenAtBoundary(FakeDetections):
        def poll(self):
            return None if port.limit_reached else super().poll()
    brain = make(port=port, source=FrozenAtBoundary())
    brain.request('phone', 'standing', max_distance_ft=.0001)
    wait_for(lambda: brain.status()['state'] == 'Stopped', timeout=4)
    assert 'no fresh camera frame' in brain.status()['message']
    assert brain.status()['last_pick'] is None
    count = len(port.commands)
    time.sleep(.15)
    assert not any(c.startswith('DRIVE ') for c in port.commands[count:])


def test_suction_mission_runs_cup_and_does_not_claim_verified_take(config, make):
    config['handoff']['grasp'] = 'suction'
    brain = make()
    brain.request('phone', 'standing', max_distance_ft=3)
    result = wait_for(lambda: brain.status()['last_pick'])
    assert result['result'] == 'unverified'
    assert result['take_verified'] is False
    assert brain.arm.object == 'hand'
    assert 'visually confirm' in brain.status()['message']


@pytest.mark.parametrize('key,value', [('grasp', 'unknown'), ('suction_hold_s', float('nan')),
                                      ('suction_dwell_s', -1), ('suction_hold_s', 61)])
def test_grasp_settings_refuse_bad_values(config, key, value):
    from brain.mission import grasp_settings
    config['handoff']['grasp'] = 'suction'
    config['handoff'][key] = value
    with pytest.raises(ValueError):
        grasp_settings(config)


def test_reapproach_requires_new_stop_before_pick(make):
    # After one acknowledged parking stop, force another approach. The next pick
    # must wait for a NEW stop, not reuse the prior stop's ticket.
    slot = {}
    class Navigation:
        phase = 0
        def search(self): return False, (0, 0)
        def approach(self, detection):
            brain = slot['brain']
            if self.phase == 0 and brain._parking_stop_ticket is not None and brain._stop_completed >= brain._parking_stop_ticket:
                self.phase = 1
                return False, (10, 10)
            if self.phase == 1:
                self.phase = 2
                slot['prior_stops'] = brain._stop_completed
            return True, (0, 0)
    def pick(*args, **kwargs):
        assert slot['brain']._stop_completed > slot['prior_stops']
        return dict(result='taken')
    brain = make(navigator_factory=Navigation, pick=pick)
    slot['brain'] = brain
    brain.request('phone', 'standing')
    result = wait_for(lambda: brain.status()['last_pick'])
    assert result['result'] == 'taken', result


@pytest.mark.parametrize('change,match', [
    ('flags', 'P4'), ('frozen', 'wrist_roll'), ('port', 'arm port'), ('cap', 'safety limit')])
def test_real_startup_rejects_dangerous_config_before_opening_devices(config, change, match):
    config['base'].update(wiring_confirmed=True, drive_calibrated=True, serial_port='/dev/cu.R4')
    if change == 'flags': config['base']['wiring_confirmed'] = 'false'
    if change == 'frozen': config['arm'].pop('frozen_joints')
    if change == 'port': config['base']['serial_port'] = config['arm']['follower_port'].replace('/dev/tty.', '/dev/cu.')
    if change == 'cap': config['arm']['max_relative_target_deg'] = float('nan')
    with pytest.raises(RuntimeError, match=match):
        build(config, sim=False)


@pytest.mark.parametrize('grasp,mode', [('claw','hand'), ('suction','suction')])
def test_real_builder_selects_matching_pose_mode(config, monkeypatch, grasp, mode):
    import serial
    import arm.poses
    import arm.real
    import perception.floor
    import brain.mission as mission
    config['base'].update(wiring_confirmed=True, drive_calibrated=True, serial_port='/dev/cu.R4')
    config['handoff']['grasp'] = grasp
    loaded = []
    monkeypatch.setattr(arm.poses, 'load', lambda path, mode: loaded.append(mode) or {})
    class Floor:
        def parking(self): return config['perception']['parking_spot_floor_m']
    monkeypatch.setattr(perception.floor.FloorMap, 'load', lambda *a, **kw: Floor())
    monkeypatch.setattr(serial, 'Serial', lambda *a, **kw: FakeSerial(config=config['base']))
    monkeypatch.setattr(mission, 'UdpDetections', lambda *a: FakeDetections())
    monkeypatch.setattr(arm.real, 'RealArm', lambda *a, **kw: SimArm())
    monkeypatch.setattr(mission, 'Mission', lambda config, **kw: kw)
    result = mission.build_real(config)
    try:
        assert loaded == [mode]
        assert result['simulated'] is False
    finally:
        result['base'].close(); result['source'].close(); result['arm'].close()


def test_extended_arm_refuses_base_motion_and_can_retry_after_manual_ready(make):
    from arm.sim import DEFAULT_POSES
    arm = SimArm()
    arm.pos.update(DEFAULT_POSES['at_object'])
    port = FakeSerial()
    brain = make(arm=arm, port=port)
    brain.request('phone', 'standing')
    wait_for(lambda: brain.status()['state'] == 'Stopped')
    assert 'ready pose' in brain.status()['message']
    assert not any(c.startswith('DRIVE ') for c in port.commands)
    # Stand-in for a person safely returning the arm; the brain never auto-moves it.
    arm.pos.update(DEFAULT_POSES['ready'])
    brain.request('phone', 'standing')
    assert wait_for(lambda: brain.status()['last_pick'])['result'] == 'taken'


class Wandering:
    """Searches forever with a nonzero drive."""
    def search(self): return False, (40, 40)
    def approach(self, detection): return False, (40, 40)


class Empty:
    def poll(self): return None
    def close(self): pass


def nonzero_drives(port):
    return [c for c in port.commands if c.startswith('DRIVE ') and c.split()[1:] != ['0', '0']]


def stopped_without_driving(brain, port, message):
    wait_for(lambda: brain.status()['state'] == 'Stopped')
    assert message in brain.status()['message']
    wait_for(lambda: port.commands[-1] == 'STOP')
    count = len(nonzero_drives(port))
    time.sleep(.4)
    assert len(nonzero_drives(port)) == count
    assert port.left == port.right == 0


def test_arm_leaving_ready_mid_search_stops_further_drive(make):
    arm, port = SimArm(), FakeSerial()
    brain = make(arm=arm, port=port, navigator_factory=Wandering, source=Empty())
    brain.request('phone', 'standing')
    wait_for(lambda: nonzero_drives(port))
    arm.pos['shoulder_lift'] += 30
    stopped_without_driving(brain, port, 'ready pose')
    assert brain._io.is_alive() and not brain._io_error


def test_dropped_retry_with_arm_off_ready_does_not_drive(make):
    arm, port = SimArm(), FakeSerial()
    calls = []
    def pick(arm_, poses, **kwargs):
        calls.append(1)
        arm.pos['elbow_flex'] += 30
        return dict(result='dropped')
    class Parked(Wandering):
        def approach(self, detection): return True, (0, 0)
    brain = make(arm=arm, port=port, navigator_factory=Parked, pick=pick)
    brain.request('phone', 'standing')
    wait_for(lambda: calls)
    stopped_without_driving(brain, port, 'ready pose')
    assert len(calls) == 1


@pytest.mark.parametrize('when', ['limit', 'drive'])
def test_unreadable_arm_stops_without_killing_io(make, when):
    class Flaky(SimArm):
        fail = when == 'limit'
        def read(self):
            if self.fail:
                raise OSError('bus timeout')
            return super().read()
    arm, port = Flaky(), FakeSerial()
    brain = make(arm=arm, port=port, navigator_factory=Wandering, source=Empty())
    brain.request('phone', 'standing')
    if when == 'drive':
        wait_for(lambda: nonzero_drives(port))
        arm.fail = True
    stopped_without_driving(brain, port, 'Arm unreadable')
    assert 'bus timeout' in brain.status()['message']
    assert brain._io.is_alive() and not brain._io_error
    brain.stop()  # Not a control-connection failure: STOP and status still work.
    assert brain.status()['state'] == 'Stopped' and 'STOP FAILED' not in brain.status()['message']


def test_stale_ready_reading_is_refreshed_before_drive(make):
    arm, port = SimArm(), FakeSerial()
    reads = []
    original = arm.read
    arm.read = lambda: reads.append(time.monotonic()) or original()
    brain = make(arm=arm, port=port, navigator_factory=Wandering, source=Empty())
    brain.request('phone', 'standing')
    wait_for(lambda: len(reads) > 3)
    gaps = [b - a for a, b in zip(reads, reads[1:])]
    assert len(reads) > 2 and max(gaps) < .35
    assert len(reads) < len(nonzero_drives(port))  # not read on every 20 Hz drive


def test_flickering_item_is_researched_at_most_three_times(make):
    slot, built = {}, []
    class Flicker(FakeDetections):
        def poll(self):
            message = super().poll()
            if slot['brain'].status()['state'] == 'Approach':
                message['items'] = []
            return message
    class Navigation(Wandering):
        def __init__(self): built.append(1)
        def approach(self, detection):
            slot['brain']._last_seen = 0  # Item lost immediately.
            return False, (10, 10)
    brain = make(navigator_factory=Navigation, source=Flicker())
    slot['brain'] = brain
    brain.request('phone', 'standing')
    before = len(built)
    wait_for(lambda: brain.status()['state'] == 'Failed', timeout=10)
    assert brain.status()['message'] == 'Item repeatedly lost; stopping'
    assert len(built) - before == 3
    brain.request('phone', 'standing')
    assert brain._researches == 0


@pytest.mark.parametrize('offset,expected', [(5.0, True), (5.1, False), (-5.0, True), (-5.1, False)])
def test_arm_ready_boundary(make, offset, expected):
    arm = SimArm()
    brain = make(arm=arm)
    arm.pos['wrist_flex'] = brain.poses['ready']['wrist_flex'] + offset
    assert brain._arm_ready() is expected


def test_arm_ready_rejects_nan_and_missing_and_ignores_frozen(make):
    arm = SimArm()
    brain = make(arm=arm)
    arm.pos['wrist_roll'] = 90.0  # frozen in robot.yaml
    arm.pos['gripper'] = 90.0  # not checked
    assert brain._arm_ready()
    arm.pos['elbow_flex'] = float('nan')
    assert not brain._arm_ready()
    del arm.pos['elbow_flex']
    assert not brain._arm_ready()


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '60', None])
def test_bad_handoff_timeout_fails_before_any_hardware(config, value):
    config['handoff']['timeout_s'] = value
    config['base'].update(wiring_confirmed=True, drive_calibrated=True, serial_port='/dev/cu.R4')
    with pytest.raises(ValueError, match='timeout_s'):
        build(config, sim=False)
    with pytest.raises(ValueError, match='timeout_s'):
        build(config, sim=True)
