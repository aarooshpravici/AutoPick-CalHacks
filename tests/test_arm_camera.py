"""A14: manual switch from the colour detector to the grey IR arm camera. No hardware."""
import copy
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from arm.sim import SimArm
from base.driver import BaseDriver, FakeSerial
from brain.arm_camera import ArmCamera, DetectorControl, FrameTap
from brain.detections import FakeDetections
from brain.mission import Mission, arm_policy_settings
from perception.publish import Released, publish, wait_for_resume
from perception.realsense import FakeCamera, Settings
from perception.publish import FakeDetector
from ui.server import create_app


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
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/robot.yaml').read_text())
    cfg['handoff']['arm_policy'] = 'data/checkpoints/example/pretrained_model'
    return cfg


class FakePublisher:
    """Answers release like perception.publish; records what it was sent."""

    def __init__(self, answer=True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(.1)
        self.received, self.answer, self.closed = [], answer, threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    @property
    def address(self):
        return "127.0.0.1:%d" % self.sock.getsockname()[1]

    def run(self):
        while not self.closed.is_set():
            try:
                data, sender = self.sock.recvfrom(64)
            except socket.timeout:
                continue
            self.received.append(data)
            if data == b"release" and self.answer:
                self.sock.sendto(b"released", sender)

    def close(self):
        self.closed.set(); self.sock.close()


class FakeCam:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def read(self):
        return np.full((18, 32, 3), 90, dtype=np.uint8)

    def close(self):
        self.log.append(f"close {self.name}")


class FakePolicy:
    def __init__(self, log):
        self.cameras = {"scene": FakeCam(log, "scene"), "wrist": FakeCam(log, "wrist")}


def test_release_waits_for_confirmation_and_resume_is_sent():
    publisher = FakePublisher()
    try:
        control = DetectorControl(publisher.address)
        control.release(timeout=1)
        control.resume()
        wait_for(lambda: publisher.received[-1:] == [b"resume"])
        assert publisher.received[0] == b"release"
    finally:
        publisher.close()


def test_release_without_publisher_fails_and_opens_no_camera(config):
    publisher = FakePublisher(answer=False)
    loaded = []
    try:
        config['perception']['control_udp'] = publisher.address
        camera = ArmCamera(config, preview=False, load_policy=lambda path: loaded.append(path))
        camera.control.release = lambda timeout=5.0: DetectorControl.release(camera.control, timeout=.3)
        with pytest.raises(RuntimeError, match="did not release"):
            camera.open(log=lambda m: None)
        assert loaded == []
    finally:
        publisher.close()


def test_open_releases_first_taps_frames_and_close_resumes(config, tmp_path):
    publisher = FakePublisher()
    log = []
    try:
        config['perception']['control_udp'] = publisher.address
        def load(path):
            log.append("load after " + publisher.received[0].decode())
            return FakePolicy(log)
        camera = ArmCamera(config, preview=False, load_policy=load)
        policy = camera.open(log=lambda m: None)
        assert log == ["load after release"]
        assert all(isinstance(cam, FrameTap) for cam in policy.cameras.values())
        assert sorted(p.name for p in camera.folder.glob("*.jpg")) == ["scene.jpg", "wrist.jpg"]
        assert camera.close() == []
        assert sorted(log[1:]) == ["close scene", "close wrist"]
        wait_for(lambda: publisher.received[-1:] == [b"resume"])
    finally:
        publisher.close()


def test_publisher_closes_camera_before_confirming_release_and_reconfirms_while_waiting():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control, \
         socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as brain, \
         socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
        control.bind(("127.0.0.1", 0)); control.setblocking(False)
        brain.bind(("127.0.0.1", 0)); brain.settimeout(1)
        brain.sendto(b"release", control.getsockname())
        time.sleep(.05)
        with pytest.raises(Released) as release:
            with FakeCamera(Settings((60, 40), (30, 20), 30, True, False)) as camera:
                publish(camera, FakeDetector(), out, brain.getsockname(), hz=1000, control=control)
        assert release.value.sender == brain.getsockname()
        brain.sendto(b"release", control.getsockname())      # a lost "released" is answered again
        brain.sendto(b"resume", control.getsockname())
        wait_for_resume(control)
        assert brain.recv(64) == b"released"


def test_arm_policy_settings_validate(config):
    assert arm_policy_settings(config).endswith('pretrained_model')
    assert arm_policy_settings(dict(config, handoff=dict(config['handoff'], arm_policy=None))) is None
    with pytest.raises(ValueError, match="claw"):
        arm_policy_settings(dict(config, handoff=dict(config['handoff'], grasp='suction')))
    with pytest.raises(ValueError):
        arm_policy_settings(dict(config, perception=dict(config['perception'], control_udp='nope')))


class RecordingCamera:
    def __init__(self, log):
        self.log = log

    def open(self, log=print, timeout=10.0):
        self.log.append("open")
        return "POLICY"

    def close(self):
        self.log.append("close")
        return []


@pytest.fixture
def make(config, tmp_path):
    brains = []
    def create(arm_camera, pick=None):
        driver = BaseDriver(FakeSerial(), max_speed_mm_s=200, max_lift_mm=1000)
        driver.status(timeout=1)
        options = dict(pick=pick) if pick else {}
        brain = Mission(copy.deepcopy(config), base=driver, arm=SimArm(), source=FakeDetections(),
                        period=.01, log_dir=tmp_path, arm_camera=arm_camera, **options)
        brains.append(brain)
        return brain
    yield create
    for brain in brains:
        brain.close()


def test_parked_mission_waits_for_the_button_then_runs_the_policy(make):
    log, picks = [], []
    def pick(arm, poses, **kwargs):
        picks.append(kwargs.get("policy")); log.append("pick")
        return dict(result="taken", grasp="policy", attempts=1, seconds=1)
    brain = make(RecordingCamera(log), pick)
    with pytest.raises(RuntimeError, match="not parked"):
        brain.arm_camera_go()
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'ArmCamera')
    time.sleep(.1)
    assert brain.status()['state'] == 'ArmCamera' and log == [] and picks == []
    brain.arm_camera_go()
    wait_for(lambda: brain.status()['last_pick'])
    assert log == ["open", "pick", "close"] and picks == ["POLICY"]


def test_camera_closes_even_when_the_pick_fails(make):
    log = []
    def pick(arm, poses, **kwargs):
        raise RuntimeError("policy blew up")
    brain = make(RecordingCamera(log), pick)
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'ArmCamera')
    brain.arm_camera_go()
    wait_for(lambda: brain.status()['last_pick'])
    assert log == ["open", "close"]
    assert brain.status()['last_pick']['result'] == 'stuck'


def test_stop_cancels_a_parked_switch(make):
    log = []
    brain = make(RecordingCamera(log))
    brain.request('phone', 'seated')
    wait_for(lambda: brain.status()['state'] == 'ArmCamera')
    brain.stop()
    with pytest.raises(RuntimeError):
        brain.arm_camera_go()
    time.sleep(.1)
    assert log == []


def test_page_button_route(make):
    brain = make(RecordingCamera([]))
    with TestClient(create_app(brain)) as client:
        assert client.post('/arm-camera').status_code == 409
        brain.request('phone', 'seated')
        wait_for(lambda: brain.status()['state'] == 'ArmCamera')
        assert client.post('/arm-camera').status_code == 200
        wait_for(lambda: brain.status()['last_pick'])
        assert 'id="armcam"' in client.get('/').text
