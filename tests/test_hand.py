"""Hand-height tests (task L3). No camera and no MediaPipe needed."""

import json
import socket
import types

import numpy as np
import pytest

from perception import hand
from perception.hand import Calibration, FakeHand, message, palm_row, publish


def lm(y):
    return types.SimpleNamespace(x=0.5, y=y, z=0.0)


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
    assert message(None, 0.0, calib, t=12.0) == {"t": 12.0, "hand_visible": False, "height_m": None, "conf": 0.0}


class FastClock:
    @staticmethod
    def monotonic():
        return 0.0

    @staticmethod
    def sleep(dt):
        pass


def test_publish_sends_udp_json():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    calib = Calibration([600, 200], [0.6, 1.0])
    detections = iter([(400, 0.9), None, (200, 0.7)])
    publish(lambda: np.zeros((720, 1280, 3), np.uint8), lambda f: next(detections), calib,
            rx.getsockname(), count=3, clock=FastClock)
    msgs = [json.loads(rx.recv(4096)) for _ in range(3)]
    rx.close()
    assert [m["hand_visible"] for m in msgs] == [True, False, True]
    assert msgs[0]["height_m"] == pytest.approx(0.8) and msgs[2]["height_m"] == pytest.approx(1.0)


def test_fake_hand_stays_in_range():
    fake = FakeHand()
    found = fake.detect(fake.read())
    assert found is not None
    assert 0.6 <= fake.calib.height(found[0]) <= 0.9


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
