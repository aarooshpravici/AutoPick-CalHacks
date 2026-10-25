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

    def __init__(self, port="/dev/ttyFAKE", motors=None, calibration=None, protocol_version=0):
        """Same signature as FeetechMotorsBus, so RealArm can rebuild it without frozen motors."""
        self.port = port
        self.motors = dict(motors) if motors is not None else {j: i + 1 for i, j in enumerate(JOINTS)}
        self.calibration = dict(calibration) if calibration is not None else {j: f"cal-{j}" for j in self.motors}
        self.protocol_version = protocol_version
        self.disconnects = []
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
        cap = self.config.max_relative_target    # like LeRobot's ensure_safe_goal_position
        if cap is not None and not isinstance(cap, (float, dict)):
            raise TypeError(cap)
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


def test_an_integer_step_cap_from_robot_yaml_is_passed_as_a_float(new_layout):
    """robot.yaml's `max_relative_target_deg: 10` loads as int; LeRobot raises TypeError(10) on it."""
    arm = RealArm("/dev/ttyFAKE", "so101_follower_arm", 10)
    assert isinstance(FakeFollower.instances[-1].config.max_relative_target, float)
    arm.command({j: 0.0 for j in JOINTS})                  # the first command no longer raises


# --- frozen joints: servo 5 (wrist_roll) is never called -----------------------------

def test_a_frozen_joint_is_never_called(new_layout):
    arm = RealArm("/dev/ttyFAKE", "so101_follower_arm", 10, frozen=["wrist_roll"])
    robot = FakeFollower.instances[-1]
    assert "wrist_roll" not in robot.bus.motors and "wrist_roll" not in robot.bus.calibration
    assert set(robot.bus.motors) == set(JOINTS) - {"wrist_roll"}
    arm.command({j: 1.0 for j in JOINTS})
    assert all(e[2] != "wrist_roll" for e in robot.bus.log if e[0] == "write")   # no register writes
    assert "wrist_roll.pos" not in robot.actions[-1]                              # no goal sent
    assert set(robot.actions[-1]) == {f"{j}.pos" for j in JOINTS if j != "wrist_roll"}


def test_a_frozen_joint_reads_as_a_constant(new_layout):
    arm = RealArm("/dev/ttyFAKE", "so101_follower_arm", 10, frozen=("wrist_roll",))
    pos = arm.read()
    assert pos["wrist_roll"] == 0.0 and set(pos) == set(JOINTS)
    assert pos["elbow_flex"] == 2.5                                               # others still real


@pytest.mark.parametrize("bad", [["gripper"], ["elbow"], ["shoulder_pan", "pinky"]])
def test_only_arm_joints_can_be_frozen(new_layout, bad):
    with pytest.raises(ValueError, match="frozen joints"):
        RealArm("/dev/ttyFAKE", "so101_follower_arm", 10, frozen=bad)


def test_nothing_frozen_keeps_the_full_bus(new_layout):
    RealArm("/dev/ttyFAKE", "so101_follower_arm", 10)
    assert set(FakeFollower.instances[-1].bus.motors) == set(JOINTS)


# --- Ctrl-C or an error during the torque ramp ----------------------------------------------

@pytest.mark.parametrize("exc", [KeyboardInterrupt, ConnectionError])
def test_an_interrupted_ramp_holds_at_the_full_caps_and_closes_the_port(new_layout, exc):
    """Stopped part-way up the ramp, the arm must not stay connected at a 20-90 % torque limit:
    every motor gets its full cap in one write, then the port closes with the torque on."""
    waits = []

    def sleep(s):
        waits.append(s)
        if len(waits) == 3:
            raise exc("stopped during the ramp")

    with pytest.raises(exc):
        RealArm("/dev/ttyFAKE", "so101_follower_arm", 10, frozen=["wrist_roll"], sleep=sleep)
    bus = FakeFollower.instances[-1].bus
    after = bus.log[bus.log.index(("enable", None)) + 1:]
    caps = {j: (500 if j in ("gripper", "shoulder_pan") else 1000) for j in JOINTS if j != "wrist_roll"}
    assert {e[2]: e[3] for e in after[-len(caps):]} == caps          # the last write: full caps, all of them
    assert {e[2]: e[3] for e in after[:len(caps)]} != caps           # it really was part-way up
    assert all(e[1] == "Torque_Limit" for e in after)                # and no goal moved
    assert bus.disconnects == [False]                                # closed, torque left on
