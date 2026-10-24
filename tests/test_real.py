"""arm/real.py against a fake LeRobot SO-101 follower. No hardware, no real lerobot import.

Guards the LeRobot layout RealArm relies on (0.6.1 moved so101_follower to so_follower)
and the safety settings it passes: never calibrate, keep torque on at exit, always a
step cap.
"""

import dataclasses
import math
import sys
import types

import pytest

from arm.poses import JOINTS
from arm.real import RealArm


@dataclasses.dataclass
class FakeConfig:
    port: str
    id: str | None = None
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = False


@dataclasses.dataclass
class OldFakeConfig:                    # a LeRobot without disable_torque_on_disconnect
    port: str
    id: str | None = None
    max_relative_target: float | None = None


class FakeBus:
    def __init__(self):
        self.disconnects = []

    def disconnect(self, disable_torque=True):
        self.disconnects.append(disable_torque)


class FakeFollower:
    instances = []
    calibrated = True

    def __init__(self, config):
        self.config = config
        self.bus = FakeBus()
        self.connect_calls = []
        self.actions = []
        self.disconnected = False
        FakeFollower.instances.append(self)

    @property
    def is_calibrated(self):
        return FakeFollower.calibrated

    def connect(self, calibrate=True):
        self.connect_calls.append(calibrate)

    def get_observation(self):
        obs = {f"{j}.pos": float(i) + 0.5 for i, j in enumerate(JOINTS)}
        obs["wrist"] = object()          # camera frames ride along in real observations
        return obs

    def send_action(self, action):
        self.actions.append(dict(action))
        return action

    def disconnect(self):
        self.disconnected = True


def install(monkeypatch, module_name, config_cls=FakeConfig):
    FakeFollower.instances = []
    FakeFollower.calibrated = True
    mod = types.ModuleType(module_name)
    mod.SO101Follower = FakeFollower
    mod.SO101FollowerConfig = config_cls
    monkeypatch.setitem(sys.modules, module_name, mod)


@pytest.fixture
def old_layout(monkeypatch):
    install(monkeypatch, "lerobot.robots.so101_follower")


@pytest.fixture
def new_layout(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.robots.so101_follower", None)   # import fails
    install(monkeypatch, "lerobot.robots.so_follower")


@pytest.mark.parametrize("layout", ["old_layout", "new_layout"])
def test_connects_without_calibrating_and_keeps_torque(request, layout):
    request.getfixturevalue(layout)
    arm = RealArm(port="/dev/fake", robot_id="so101_follower_arm", max_relative_target=10)
    robot = FakeFollower.instances[-1]
    assert robot.config == FakeConfig(port="/dev/fake", id="so101_follower_arm", max_relative_target=10,
                                      disable_torque_on_disconnect=False, use_degrees=True)
    assert robot.connect_calls == [False]

    state = arm.read()
    assert list(state) == JOINTS and all(isinstance(v, float) for v in state.values())

    arm.command({"shoulder_pan": 3, "gripper": 140.0, "not_a_joint": 1})
    arm.command({"gripper": -7})
    assert robot.actions == [{"shoulder_pan.pos": 3.0, "gripper.pos": 100.0}, {"gripper.pos": 0.0}]

    arm.close()
    assert robot.disconnected and robot.bus.disconnects == []


def test_old_lerobot_closes_bus_with_torque_on(monkeypatch):
    install(monkeypatch, "lerobot.robots.so101_follower", OldFakeConfig)
    arm = RealArm(port="/dev/fake", robot_id="so101_follower_arm", max_relative_target=10)
    robot = FakeFollower.instances[-1]
    arm.close()
    assert not robot.disconnected and robot.bus.disconnects == [False]


def test_refuses_without_matching_calibration(old_layout):
    FakeFollower.calibrated = False
    with pytest.raises(RuntimeError, match="calibration"):
        RealArm(port="/dev/fake", robot_id="so101_follower_arm", max_relative_target=10)
    assert FakeFollower.instances[-1].bus.disconnects == [False]


@pytest.mark.parametrize("cap", [None, 0, -5, math.nan, math.inf, True, "10"])
def test_requires_a_step_cap(old_layout, cap):
    """None would hand LeRobot 'no cap at all'; refuse before anything is opened."""
    with pytest.raises(ValueError, match="max_relative_target"):
        RealArm(port="/dev/fake", robot_id="so101_follower_arm", max_relative_target=cap)
    assert FakeFollower.instances == []


def test_step_cap_has_no_default():
    with pytest.raises(TypeError):
        RealArm(port="/dev/fake", robot_id="so101_follower_arm")
