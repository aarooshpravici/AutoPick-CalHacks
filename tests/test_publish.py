import json
import socket

import numpy as np
import pytest

from perception.publish import Detection, FakeDetector, load_homography, make_message, publish
from perception.realsense import FakeCamera, Frame, Settings
from perception.listen import summary


@pytest.fixture
def frame():
    return Frame(np.zeros((4, 6, 3), dtype=np.uint8),
                 np.array([[0, 100, 200, 300, 0, 0]] * 4), .001, 123., 7, True)


def test_message_schema_depth_and_bottom_centre(frame):
    result = make_message(frame, [Detection("phone", .9, (0, 0, 4, 4))],
                          homography=np.diag([.1, .2, 1]))
    assert result == {"t": 123., "frame": 7, "items": [dict(
        label="phone", conf=.9, bbox_px=[0, 0, 4, 4], centre_px=[2, 2],
        floor_m=[.2, .8], depth_m=.2)]}
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("aligned,depth,scale", [
    (False, np.ones((4, 6)), .001), (True, np.zeros((4, 6)), .001),
    (True, np.ones((2, 3)), .001), (True, np.ones((4, 6)), float("nan")),
])
def test_missing_depth_is_null(frame, aligned, depth, scale):
    frame.aligned, frame.depth, frame.depth_scale = aligned, depth, scale
    item = make_message(frame, [Detection("phone", .9, (0, 0, 4, 4))])["items"][0]
    assert item["depth_m"] is None
    assert item["floor_m"] is None


def test_empty_and_invalid_detections(frame):
    assert make_message(frame, [])["items"] == []
    boxes = [Detection("phone", .49, (0, 0, 4, 4)),
             Detection("phone", float("nan"), (0, 0, 4, 4)),
             Detection("phone", .9, (4, 0, 0, 4)),
             Detection("phone", .9, (0, 0, float("inf"), 4))]
    assert make_message(frame, boxes)["items"] == []


def test_clipped_box_depth_ignores_invalid_values(frame):
    frame.depth = np.full((4, 6), np.nan)
    frame.depth[0, :4] = [-1, 0, 100, 300]
    item = make_message(frame, [Detection("phone", .9, (-10, -1, 10, 8))])["items"][0]
    assert item["bbox_px"] == [0, 0, 6, 4]
    assert item["depth_m"] == .2


def test_homography_file_validates_camera_geometry(tmp_path):
    path = tmp_path / "floor_homography.json"
    settings = Settings((6, 4), (3, 2), 30, True, False)
    assert load_homography(path, settings) is None
    path.write_text(json.dumps(dict(matrix=np.eye(3).tolist(), colour_size=[6, 4], flip=False)))
    np.testing.assert_array_equal(load_homography(path, settings), np.eye(3))
    with pytest.raises(ValueError, match="size/flip"):
        load_homography(path, Settings((6, 4), (3, 2), 30, True, True))


def test_udp_publish_uses_stub_without_weights():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            with FakeCamera(Settings((60, 40), (30, 20), 30, True, False)) as camera:
                assert publish(camera, FakeDetector(), sender, receiver.getsockname(),
                               count=1, hz=1000) == 1
        message = json.loads(receiver.recv(65535))
    assert message["frame"] == 1
    assert message["items"][0]["label"] == "phone"
    assert message["items"][0]["depth_m"] == 1
    assert message["items"][0]["floor_m"] is None


def test_live_summary_includes_empty_frames(frame):
    message = make_message(frame, [Detection("phone", .9, (0, 0, 4, 4))])
    result = summary([message, make_message(frame, [])], 1, "0.5m-left")
    assert result == dict(label="phone", placement="0.5m-left", frames=2, detected_frames=1,
                          detected=True, hz=2, max_conf=.9, mean_pixel_height=4)
