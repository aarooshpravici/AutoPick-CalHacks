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
    """Records register writes and torque-enables in order, like the Feetech bus would see them."""

    def __init__(self):
        self.disconnects = []
        self.motors = {j: i + 1 for i, j in enumerate(JOINTS)}
        self.present = {j: 1800 + 37 * i for i, j in enumerate(JOINTS)}   # raw ticks, not a stale goal
        self.log = []
        self.is_connected = False

    def disconnect(self, disable_torque=True):
        self.disconnects.append(disable_torque)
        self.is_connected = False

    def sync_read(self, register, motors=None, normalize=True, num_retry=0):
        assert register == "Present_Position" and normalize is False
        return {m: self.present[m] for m in (list(self.motors) if motors is None else motors)}

    def write(self, register, motor, value, normalize=True, num_retry=0):
        assert normalize is False
        self.log.append(("write", register, motor, value))

    def sync_write(self, register, values, normalize=True, num_retry=0):
        assert normalize is False
        for motor, value in values.items():
            self.log.append(("write", register, motor, value))

    def enable_torque(self, motors=None, num_retry=0):
        self.log.append(("enable", motors))


class FakeFollower:
    instances = []
    calibrated = True
    fail_connects = 0                    # handshakes that fail before one succeeds

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
        """Like LeRobot 0.6.2: open the bus, then configure() re-enables the torque."""
        self.connect_calls.append(calibrate)
        self.bus.is_connected = True
        if FakeFollower.fail_connects > 0:
            FakeFollower.fail_connects -= 1
            raise ConnectionError("Failed to write 'Lock' on id_=3 with '1' after 1 tries.\n[TxRxResult] ...")
        self.bus.enable_torque()

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
    FakeFollower.fail_connects = 0
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


# --- no startup jolt (same fix as safe_teleop.py) -----------------------------------

@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    import arm.real as real_mod
    monkeypatch.setattr(real_mod, "time", types.SimpleNamespace(sleep=lambda s: None), raising=False)


def test_goal_is_set_to_present_before_the_torque_comes_on(new_layout):
    RealArm("/dev/ttyFAKE", "so101_follower_arm", 10)
    bus = FakeFollower.instances[-1].bus
    enable_at = bus.log.index(("enable", None))
    before = bus.log[:enable_at]
    for j in JOINTS:
        assert ("write", "Goal_Position", j, bus.present[j]) in before
        cap = 500 if j in ("gripper", "shoulder_pan") else 1000
        assert ("write", "Torque_Limit", j, int(cap * 0.2)) in before


def test_torque_limit_ramps_up_to_the_caps_after_connect(new_layout):
    waits = []
    RealArm("/dev/ttyFAKE", "so101_follower_arm", 10, ramp_s=1.5, sleep=waits.append)
    bus = FakeFollower.instances[-1].bus
    after = bus.log[bus.log.index(("enable", None)) + 1:]
    assert all(e[0] == "write" and e[1] == "Torque_Limit" for e in after)   # no Goal moves while ramping
    last = {e[2]: e[3] for e in after}
    assert last == {j: (500 if j in ("gripper", "shoulder_pan") else 1000) for j in JOINTS}
    first = {e[2]: e[3] for e in after[:len(JOINTS)]}
    assert all(first[j] < last[j] for j in JOINTS)                          # it rises, not jumps
    assert sum(waits) == pytest.approx(1.5)


def test_a_missed_handshake_is_retried(new_layout):
    FakeFollower.fail_connects = 2
    RealArm("/dev/ttyFAKE", "so101_follower_arm", 10)
    robot = FakeFollower.instances[-1]
    assert robot.connect_calls == [False, False, False]
    assert robot.bus.disconnects == [False, False]                          # closed between tries
    assert ("enable", None) in robot.bus.log


def test_gives_up_after_five_failed_handshakes_without_enabling_torque(new_layout):
    FakeFollower.fail_connects = 5
    with pytest.raises(RuntimeError, match="not answering after 5 tries"):
        RealArm("/dev/ttyFAKE", "so101_follower_arm", 10)
    assert ("enable", None) not in FakeFollower.instances[-1].bus.log
