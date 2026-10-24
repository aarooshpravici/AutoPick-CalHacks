"""Hand-height tests (task L3). MediaPipe and the camera are faked; nothing opens hardware."""

import copy
import json
import socket
import sys
import types

import numpy as np
import pytest
import yaml

from perception import hand
from perception.hand import (Calibration, ConfigError, FakeHand, camera_index, message, palm_row, presets_mm,
                             publish, tray_target_mm, udp_target)

PLAN_9_3_KEYS = ["t", "hand_visible", "height_m", "conf"]

# Same keys as config/robot.yaml from task S6 (with the new arm ids).
ROBOT = {
    "arm": {"follower_port": "/dev/tty.usbmodemXXXX", "leader_port": "/dev/tty.usbmodemYYYY",
            "follower_id": "so101_follower_arm", "leader_id": "so101_leader_arm",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 1, "scene": 2, "mast_top": 3},
    "realsense": {"colour": [1280, 720], "depth": [848, 480], "fps": 30, "align_depth_to_colour": True,
                  "flip": False},
    "base": {"serial_port": "/dev/tty.usbserial-XXXX", "baud": 115200, "wheel_diameter_mm": 72,
             "track_width_mm": 260, "max_speed_mm_s": 200},
    "perception": {"detections_udp": "127.0.0.1:5601", "hand_udp": "127.0.0.1:5602",
                   "parking_spot_floor_m": [0.22, 0.0], "parking_tolerance_m": 0.02},
    "lift": {"seated_hand_mm": 700, "standing_hand_mm": 1000, "taken_weight_drop_g": 50},
    "items": ["phone", "glasses", "pill_bottle", "remote", "pen", "sock"],
}


def robot_cfg(**changes):
    """A copy of ROBOT; changes are {"section__key": value}, value None deletes the key."""
    cfg = copy.deepcopy(ROBOT)
    for dotted, value in changes.items():
        section, key = dotted.split("__")
        if value is None:
            cfg[section].pop(key)
        else:
            cfg[section][key] = value
    return cfg


def write_config(tmp_path, cfg):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def lm(y):
    return types.SimpleNamespace(x=0.5, y=y, z=0.0)


def hand_landmarks(palm_y):
    return [lm(palm_y)] * 21


# --- fakes for MediaPipe and OpenCV ----------------------------------------------

def _forbidden_module(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"test tried to use the real {name}")

    mod = types.ModuleType(name)
    mod.VideoCapture = refuse
    mod.Image = refuse
    mod.tasks = types.SimpleNamespace(vision=types.SimpleNamespace(
        HandLandmarker=types.SimpleNamespace(create_from_model_path=refuse)))
    return mod


@pytest.fixture(autouse=True)
def no_camera_or_mediapipe(monkeypatch):
    """Every test starts with cv2 and mediapipe replaced by modules that refuse to run."""
    monkeypatch.setitem(sys.modules, "cv2", _forbidden_module("cv2"))
    monkeypatch.setitem(sys.modules, "mediapipe", _forbidden_module("mediapipe"))


def fake_cv2(monkeypatch, frame=None, opened=True):
    log = {"opened": [], "released": 0}

    class VideoCapture:
        def __init__(self, index):
            log["opened"].append(index)

        def isOpened(self):
            return opened

        def read(self):
            return (frame is not None, frame)

        def release(self):
            log["released"] += 1

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 4
    cv2.VideoCapture = VideoCapture
    cv2.cvtColor = lambda img, code: img[..., ::-1].copy()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return log


def fake_mediapipe(monkeypatch, result_for):
    """result_for(image) -> list of (palm_y, score) hands the fake detector reports."""
    log = {}

    class Detector:
        closed = False

        def detect(self, image):
            log.setdefault("images", []).append(image)
            hands = result_for(image)
            return types.SimpleNamespace(hand_landmarks=[hand_landmarks(y) for y, _ in hands],
                                         handedness=[[types.SimpleNamespace(score=s)] for _, s in hands])

        def close(self):
            self.closed = True

    def create_from_model_path(path):
        log["path"] = path
        log["detector"] = Detector()
        return log["detector"]

    mp = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(vision=types.SimpleNamespace(
        HandLandmarker=types.SimpleNamespace(create_from_model_path=create_from_model_path)))
    mp.ImageFormat = types.SimpleNamespace(SRGB="SRGB")
    mp.Image = lambda image_format, data: types.SimpleNamespace(image_format=image_format, data=data)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    return log


def loopback_receiver():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    return rx


class FastClock:
    @staticmethod
    def monotonic():
        return 0.0

    @staticmethod
    def sleep(dt):
        pass


# --- palm row, calibration, message -----------------------------------------------

def test_palm_row_uses_wrist_and_knuckles():
    landmarks = [lm(0.0)] * 21
    for i, y in zip(hand.PALM_LANDMARKS, (0.40, 0.50, 0.50, 0.60)):
        landmarks[i] = lm(y)
    assert palm_row(landmarks, 720) == pytest.approx(0.5 * 720)


def test_calibration_fit_and_roundtrip(tmp_path):
    calib = Calibration([600, 400, 200], [0.6, 0.8, 1.0])      # higher hand = smaller row
    assert calib.height(500) == pytest.approx(0.7)
    path = tmp_path / "hand_calibration.json"
    calib.save(path)
    assert Calibration.load(path).height(300) == pytest.approx(0.9)


@pytest.mark.parametrize("rows, heights", [([100], [0.5]), ([100, 105], [0.5, 0.6]), ([1, 2], [0.5])])
def test_calibration_rejects_bad_points(rows, heights):
    with pytest.raises(ValueError):
        Calibration(rows, heights)


def test_message_matches_plan_9_3():
    calib = Calibration([600, 200], [0.6, 1.0])
    seen = message(400, 0.8234, calib, t=12.0)
    assert seen == {"t": 12.0, "hand_visible": True, "height_m": 0.8, "conf": 0.823}
    assert list(seen) == PLAN_9_3_KEYS
    assert message(None, 0.0, calib, t=12.0) == {"t": 12.0, "hand_visible": False, "height_m": None, "conf": 0.0}
    sample = json.loads('{"t": 1767225600.456, "hand_visible": true, "height_m": 0.74, "conf": 0.82}')
    real = json.loads(json.dumps(message(400, 0.82, calib)))
    assert list(real) == list(sample)
    assert all(type(real[k]) is type(sample[k]) for k in sample)


def test_message_with_a_hand_needs_a_calibration():
    # A visible hand must never go out as {"hand_visible": true, "height_m": null}.
    with pytest.raises(ValueError, match="calibration"):
        message(400, 0.9, None)
    assert message(None, 0.0, None, t=1.0)["hand_visible"] is False


# --- robot.yaml ---------------------------------------------------------------------

def test_settings_come_from_robot_yaml():
    cfg = robot_cfg()
    assert udp_target(cfg) == ("127.0.0.1", 5602)
    assert camera_index(cfg) == 3
    assert presets_mm(cfg) == {"seated": 700.0, "standing": 1000.0}
    other = robot_cfg(perception__hand_udp="10.0.0.5:6000", cameras__mast_top=0)
    assert udp_target(other) == ("10.0.0.5", 6000)
    assert camera_index(other) == 0


@pytest.mark.parametrize("changes", [
    {"perception__hand_udp": None}, {"perception__hand_udp": "127.0.0.1"}, {"perception__hand_udp": "host:0"},
    {"perception__hand_udp": ":5602"}, {"perception__hand_udp": "127.0.0.1:port"},
])
def test_udp_target_has_no_fallback(changes):
    with pytest.raises(ConfigError):
        udp_target(robot_cfg(**changes))


@pytest.mark.parametrize("value", [None, "three", True, -1, 2.5])
def test_camera_index_has_no_fallback(value):
    with pytest.raises(ConfigError):
        camera_index(robot_cfg(cameras__mast_top=value))


@pytest.mark.parametrize("changes", [
    {"lift__seated_hand_mm": None}, {"lift__standing_hand_mm": None}, {"lift__seated_hand_mm": "700"},
    {"lift__seated_hand_mm": True}, {"lift__seated_hand_mm": 1000}, {"lift__seated_hand_mm": 0},
    {"lift__standing_hand_mm": float("nan")},
])
def test_presets_have_no_fallback(changes):
    with pytest.raises(ConfigError):
        presets_mm(robot_cfg(**changes))


def test_load_config_requires_the_file(tmp_path):
    with pytest.raises(ConfigError, match="robot.yaml"):
        hand.load_config(tmp_path / "robot.yaml")
    (tmp_path / "empty.yaml").write_text("")
    with pytest.raises(ConfigError):
        hand.load_config(tmp_path / "empty.yaml")
    assert hand.load_config(write_config(tmp_path, ROBOT)) == ROBOT


# --- tray height (plan 10.6) --------------------------------------------------------

def visible(height_m):
    return {"t": 1.0, "hand_visible": True, "height_m": height_m, "conf": 0.9}


def test_v1_tray_target_is_the_preset():
    cfg = robot_cfg()
    assert tray_target_mm(cfg, "seated") == 700
    assert tray_target_mm(cfg, "standing") == 1000
    with pytest.raises(ValueError):
        tray_target_mm(cfg, "kneeling")


@pytest.mark.parametrize("height_m, expected", [(0.85, 850), (0.40, 700), (1.60, 1000), (-3.0, 700), (0.7, 700)])
def test_v2_tray_target_is_the_hand_height_within_the_presets(height_m, expected):
    for preset in hand.PRESETS:
        assert tray_target_mm(robot_cfg(), preset, visible(height_m)) == pytest.approx(expected)


@pytest.mark.parametrize("msg", [
    None, {}, {"t": 1.0, "hand_visible": False, "height_m": None, "conf": 0.0}, visible(None),
    visible(float("nan")), visible("0.8"), {"t": 1.0, "hand_visible": "yes", "height_m": 0.8, "conf": 0.9},
])
def test_v2_tray_target_without_a_hand_uses_the_preset(msg):
    assert tray_target_mm(robot_cfg(), "seated", msg) == 700
    assert tray_target_mm(robot_cfg(), "standing", msg) == 1000


def test_tray_limits_come_from_robot_yaml():
    cfg = robot_cfg(lift__seated_hand_mm=650, lift__standing_hand_mm=1200)
    assert tray_target_mm(cfg, "seated", visible(0.5)) == 650
    assert tray_target_mm(cfg, "seated", visible(1.5)) == 1200


# --- publishing ---------------------------------------------------------------------

def test_publish_sends_each_message():
    calib = Calibration([600, 200], [0.6, 1.0])
    detections = iter([(400, 0.9), None, (200, 0.7)])
    frames = iter([np.zeros((720, 1280, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8), None])
    sent = []
    publish(lambda: next(frames), lambda f: next(detections), calib, sent.append, count=3, clock=FastClock)
    assert [m["hand_visible"] for m in sent] == [True, False, False]     # no frame = no hand
    assert sent[0]["height_m"] == pytest.approx(0.8)
    assert all(list(m) == PLAN_9_3_KEYS for m in sent)


def test_udp_sender_sends_json_datagrams():
    rx = loopback_receiver()
    sender = hand.UdpSender(rx.getsockname())
    try:
        sender({"t": 1.0, "hand_visible": False, "height_m": None, "conf": 0.0})
        assert json.loads(rx.recv(4096)) == {"t": 1.0, "hand_visible": False, "height_m": None, "conf": 0.0}
    finally:
        sender.close()
        rx.close()


def test_fake_hand_stays_in_range():
    fake = FakeHand()
    found = fake.detect(fake.read())
    assert found is not None
    assert 0.6 <= fake.calib.height(found[0]) <= 0.9


# --- MediaPipe and camera wrappers (faked) ------------------------------------------

def test_mediapipe_hands_picks_the_most_confident_hand(tmp_path, monkeypatch):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"fake")
    hands_seen = [[(0.25, 0.6), (0.5, 0.9)]]
    mp = fake_mediapipe(monkeypatch, lambda image: hands_seen[0])
    hands = hand.MediaPipeHands(model)
    assert mp["path"] == str(model)
    rgb = np.zeros((720, 1280, 3), np.uint8)
    row, conf = hands.detect(rgb)
    assert row == pytest.approx(360) and conf == pytest.approx(0.9)
    assert mp["images"][0].image_format == "SRGB" and mp["images"][0].data is rgb
    hands_seen[0] = []
    assert hands.detect(rgb) is None
    hands.close()
    assert mp["detector"].closed


def test_mediapipe_hands_missing_model_explains_the_download(tmp_path):
    with pytest.raises(SystemExit, match="curl -L -o"):
        hand.MediaPipeHands(tmp_path / "missing.task")      # the refusing fake proves mediapipe isn't touched


def test_webcam_reads_rgb_frames(monkeypatch):
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 255                                         # blue in BGR
    cams = fake_cv2(monkeypatch, frame=bgr)
    cam = hand.Webcam(5)
    assert cams["opened"] == [5]
    assert cam.read()[0, 0].tolist() == [0, 0, 255]           # blue in RGB
    cam.close()
    assert cams["released"] == 1


def test_webcam_read_failure_and_open_failure(monkeypatch):
    fake_cv2(monkeypatch, frame=None)
    assert hand.Webcam(1).read() is None
    cams = fake_cv2(monkeypatch, opened=False)
    with pytest.raises(SystemExit, match="cameras.mast_top"):
        hand.Webcam(4)
    assert cams["released"] == 1


# --- calibrate ----------------------------------------------------------------------

def test_calibrate_flow():
    rows = {"600": 600.0, "800": 400.0, "1000": 200.0}
    answers = iter(["600", "oops", "800", "1000", ""])
    current = {"row": None}

    def ask(prompt):
        a = next(answers)
        current["row"] = rows.get(a)
        return a

    cam = types.SimpleNamespace(read=lambda: np.zeros((720, 1280, 3), np.uint8))
    hands = types.SimpleNamespace(detect=lambda f: (current["row"], 0.9))
    calib = hand.calibrate(cam, hands, ask=ask)
    assert calib.height(300) == pytest.approx(0.9)


def test_calibrate_needs_three_heights():
    answers = iter(["600", ""])
    cam = types.SimpleNamespace(read=lambda: np.zeros((10, 10, 3), np.uint8))
    hands = types.SimpleNamespace(detect=lambda f: (100.0, 0.9))
    with pytest.raises(SystemExit):
        hand.calibrate(cam, hands, ask=lambda p: next(answers))


def test_calibrate_rows_too_close_is_a_clean_exit():
    answers = iter(["600", "800", "1000", ""])
    cam = types.SimpleNamespace(read=lambda: np.zeros((10, 10, 3), np.uint8))
    hands = types.SimpleNamespace(detect=lambda f: (100.0, 0.9))   # same row at every height
    with pytest.raises(SystemExit, match="nothing saved"):
        hand.calibrate(cam, hands, ask=lambda p: next(answers))


# --- command line -------------------------------------------------------------------

@pytest.fixture
def files(tmp_path):
    calib = tmp_path / "hand_calibration.json"
    Calibration([600, 200], [0.6, 1.0]).save(calib)
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"fake")
    return types.SimpleNamespace(calib=calib, model=model)


def cli(cmd, config, files, *extra):
    return [cmd, "--config", str(config), "--calibration", str(files.calib), "--model", str(files.model), *extra]


def test_help_opens_nothing(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hand.main(["--help"])
    assert exit_info.value.code == 0
    assert "robot.yaml" in capsys.readouterr().out


def test_dry_run_opens_no_camera_and_no_socket(tmp_path, files, monkeypatch, capsys):
    def no_socket(*args, **kwargs):
        raise AssertionError("dry run opened a socket")

    monkeypatch.setattr(hand.socket, "socket", no_socket)
    config = write_config(tmp_path, robot_cfg())
    assert hand.main(cli("run", config, files, "--dry-run", "--count", "3")) == 0
    out, err = capsys.readouterr()
    msgs = [json.loads(line) for line in out.splitlines()]
    assert len(msgs) == 3 and all(list(m) == PLAN_9_3_KEYS for m in msgs)
    assert "camera: 3" in err and "127.0.0.1:5602" in err and "seated 700 mm" in err

    files.model.unlink()
    assert hand.main(cli("calibrate", config, files, "--dry-run")) == 0
    err = capsys.readouterr().err
    assert "camera: 3" in err and hand.MODEL_URL in err


def test_dry_run_still_checks_robot_yaml(tmp_path, files):
    config = write_config(tmp_path, robot_cfg(lift__standing_hand_mm=None))
    with pytest.raises(SystemExit, match="lift.standing_hand_mm"):
        hand.main(cli("run", config, files, "--dry-run", "--count", "1"))


@pytest.mark.parametrize("changes, missing", [
    ({"cameras__mast_top": None}, "cameras.mast_top"),
    ({"perception__hand_udp": None}, "perception.hand_udp"),
    ({"perception__hand_udp": "127.0.0.1"}, "perception.hand_udp"),
])
def test_cli_refuses_incomplete_robot_yaml_before_opening_anything(tmp_path, files, monkeypatch, changes, missing):
    cams = fake_cv2(monkeypatch, frame=np.zeros((4, 4, 3), np.uint8))
    config = write_config(tmp_path, robot_cfg(**changes))
    for argv in (cli("run", config, files, "--count", "1"), cli("run", config, files, "--fake", "--count", "1"),
                 cli("calibrate", config, files)):
        with pytest.raises(SystemExit, match=missing):
            hand.main(argv)
    assert cams["opened"] == []


def test_cli_without_robot_yaml_has_no_fallback(tmp_path, files):
    with pytest.raises(SystemExit, match="robot.yaml"):
        hand.main(cli("run", tmp_path / "robot.yaml", files, "--fake", "--count", "1"))


def test_run_checks_calibration_and_model_before_opening_the_camera(tmp_path, files, monkeypatch):
    cams = fake_cv2(monkeypatch, frame=np.zeros((4, 4, 3), np.uint8))
    config = write_config(tmp_path, robot_cfg())
    files.calib.unlink()
    with pytest.raises(SystemExit, match="calibrate"):
        hand.main(cli("run", config, files, "--count", "1"))
    files.model.unlink()
    with pytest.raises(SystemExit, match="hand model missing"):
        hand.main(cli("calibrate", config, files))
    assert cams["opened"] == []


def test_run_fake_sends_to_the_configured_udp_port(tmp_path, files):
    rx = loopback_receiver()
    config = write_config(tmp_path, robot_cfg(perception__hand_udp=f"127.0.0.1:{rx.getsockname()[1]}"))
    try:
        assert hand.main(cli("run", config, files, "--fake", "--count", "2")) == 0
        msgs = [json.loads(rx.recv(4096)) for _ in range(2)]
    finally:
        rx.close()
    assert all(list(m) == PLAN_9_3_KEYS for m in msgs)


def test_run_reads_the_mast_camera_and_sends_hand_heights(tmp_path, files, monkeypatch):
    rx = loopback_receiver()
    config = write_config(tmp_path, robot_cfg(cameras__mast_top=7,
                                              perception__hand_udp=f"127.0.0.1:{rx.getsockname()[1]}"))
    cams = fake_cv2(monkeypatch, frame=np.zeros((720, 1280, 3), np.uint8))
    mp = fake_mediapipe(monkeypatch, lambda image: [(400 / 720, 0.9)])   # palm at row 400 -> 0.8 m
    try:
        assert hand.main(cli("run", config, files, "--count", "2")) == 0
        msgs = [json.loads(rx.recv(4096)) for _ in range(2)]
    finally:
        rx.close()
    assert cams["opened"] == [7] and cams["released"] == 1
    assert mp["path"] == str(files.model) and mp["detector"].closed
    for m in msgs:
        assert list(m) == PLAN_9_3_KEYS
        assert m["hand_visible"] is True and m["height_m"] == pytest.approx(0.8) and m["conf"] == pytest.approx(0.9)


def test_calibrate_command_saves_the_fit(tmp_path, files, monkeypatch):
    config = write_config(tmp_path, robot_cfg())
    files.calib.unlink()
    rows = {"600": 600.0, "800": 400.0, "1000": 200.0}
    answers = iter(["600", "800", "1000", ""])
    current = {"row": 0.0}

    def fake_input(prompt):
        a = next(answers)
        current["row"] = rows.get(a, 0.0)
        return a

    monkeypatch.setattr("builtins.input", fake_input)
    cams = fake_cv2(monkeypatch, frame=np.zeros((720, 1280, 3), np.uint8))
    fake_mediapipe(monkeypatch, lambda image: [(current["row"] / 720, 0.9)])
    assert hand.main(cli("calibrate", config, files)) == 0
    assert Calibration.load(files.calib).height(300) == pytest.approx(0.9)
    assert cams["opened"] == [3] and cams["released"] == 1


def test_count_must_be_positive(tmp_path, files):
    config = write_config(tmp_path, robot_cfg())
    with pytest.raises(SystemExit) as exit_info:
        hand.main(cli("run", config, files, "--dry-run", "--count", "0"))
    assert exit_info.value.code == 2
