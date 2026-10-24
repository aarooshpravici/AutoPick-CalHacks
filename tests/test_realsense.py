from dataclasses import replace
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from perception.realsense import FakeCamera, RealSense, Settings, retry, snapshot, return_ownership


@pytest.fixture
def settings():
    return Settings((4, 3), (2, 2), 30, True, False)


@pytest.fixture
def sdk():
    sdk = MagicMock()
    pipe = sdk.pipeline.return_value
    profile = pipe.get_active_profile.return_value
    profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.002
    frames = pipe.wait_for_frames.return_value
    sdk.align.return_value.process.return_value = frames
    frames.get_color_frame.return_value.get_data.return_value = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    frames.get_color_frame.return_value.get_frame_number.return_value = 42
    frames.get_depth_frame.return_value.get_data.return_value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    return sdk


def test_retries_five_times_four_sleeps_and_preserves_error():
    error = RuntimeError("power state")
    call = MagicMock(side_effect=error)
    sleep = MagicMock()
    with pytest.raises(RuntimeError) as raised:
        retry(call, sleep=sleep)
    assert raised.value is error
    assert call.call_count == 5
    assert [args.args for args in sleep.call_args_list] == [(1,)] * 4


def test_does_not_retry_programming_errors():
    call = MagicMock(side_effect=ValueError("invalid configuration"))
    with pytest.raises(ValueError):
        retry(call)
    assert call.call_count == 1


def test_real_reader_retries_start_and_frames_and_owns_data(settings, sdk):
    pipe = sdk.pipeline.return_value
    frames = pipe.wait_for_frames.return_value
    pipe.start.side_effect = [RuntimeError("power state"), MagicMock()]
    pipe.wait_for_frames.side_effect = [RuntimeError("timeout"), frames]
    sleep = MagicMock()
    with RealSense(settings, sdk=sdk, sleep=sleep) as camera:
        frame = camera.read()
        assert frame.number == 42 and frame.depth_scale == 0.002 and frame.aligned
        assert frame.t > 0
        assert pipe.start.call_count == 2
        assert pipe.wait_for_frames.call_count == 2
        sdk.align.assert_called_once_with(sdk.stream.color)
        assert not np.shares_memory(frame.depth, frames.get_depth_frame().get_data())
    pipe.stop.assert_called_once()
    assert sleep.call_count == 2


def test_stream_configuration(settings, sdk):
    with RealSense(settings, sdk=sdk):
        pass
    cfg = sdk.config.return_value
    cfg.enable_stream.assert_any_call(sdk.stream.color, 4, 3, sdk.format.bgr8, 30)
    cfg.enable_stream.assert_any_call(sdk.stream.depth, 2, 2, sdk.format.z16, 30)


def test_missing_frame_retried_and_persistent_failure_closes(settings, sdk):
    frames = sdk.pipeline.return_value.wait_for_frames.return_value
    frames.get_depth_frame.return_value = None
    with pytest.raises(RuntimeError, match="missing"):
        with RealSense(settings, sdk=sdk, sleep=lambda _: None) as camera:
            camera.read()
    assert sdk.pipeline.return_value.wait_for_frames.call_count == 5
    sdk.pipeline.return_value.stop.assert_called_once()


def test_setup_failure_closes_started_pipeline(settings, sdk):
    sensor = sdk.pipeline.return_value.get_active_profile().get_device().first_depth_sensor()
    sensor.get_depth_scale.side_effect = RuntimeError("sensor unavailable")
    with pytest.raises(RuntimeError, match="sensor unavailable"):
        with RealSense(settings, sdk=sdk, sleep=lambda _: None):
            pass
    sdk.pipeline.return_value.stop.assert_called_once()


def test_flip_both_streams(settings, sdk):
    frames = sdk.pipeline.return_value.wait_for_frames.return_value
    with RealSense(replace(settings, flip=True), sdk=sdk) as camera:
        frame = camera.read()
    np.testing.assert_array_equal(frame.colour, frames.get_color_frame().get_data()[::-1, ::-1])
    np.testing.assert_array_equal(frame.depth, frames.get_depth_frame().get_data()[::-1, ::-1])


def test_unaligned_does_not_call_aligner(settings, sdk):
    with RealSense(replace(settings, align_depth_to_colour=False), sdk=sdk) as camera:
        assert camera.read().aligned is False
    sdk.align.assert_not_called()


def test_fake_dimensions_flip_and_counter(settings):
    with FakeCamera(settings) as camera:
        original = camera.read()
        assert camera.read().number == 2
    with FakeCamera(replace(settings, flip=True)) as camera:
        flipped = camera.read()
    np.testing.assert_array_equal(flipped.colour, original.colour[::-1, ::-1])
    np.testing.assert_array_equal(flipped.depth, original.depth[::-1, ::-1])
    assert flipped.depth.shape == (3, 4)
    with FakeCamera(replace(settings, align_depth_to_colour=False)) as camera:
        assert camera.read().depth.shape == (2, 2)


def test_snapshot_roundtrip_no_overwrite(settings, tmp_path):
    import cv2
    with FakeCamera(settings) as camera:
        frame = camera.read()
    output = tmp_path / "nested" / "photos"
    first = snapshot(frame, output, simulated=True)
    second = snapshot(frame, output, simulated=True)
    assert first != second
    np.testing.assert_array_equal(cv2.imread(str(first)), frame.colour)
    depth = cv2.imread(str(first.with_name(first.stem + "_depth.png")), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(depth, frame.depth)
    assert depth.dtype == np.uint16
    metadata = json.loads(first.with_suffix(".json").read_text())
    assert metadata["depth_scale_m"] == 0.001 and metadata["simulated"] is True


def test_sudo_returns_ownership(monkeypatch, tmp_path):
    chown = MagicMock()
    monkeypatch.setattr("os.chown", chown)
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "501")
    monkeypatch.setenv("SUDO_GID", "20")
    return_ownership(tmp_path)
    chown.assert_called_once_with(tmp_path, 501, 20)


@pytest.mark.parametrize("change", [{"fps": 0}, {"fps": True}, {"colour": (0, 2)}, {"flip": "false"}])
def test_reject_bad_settings(settings, change):
    with pytest.raises(ValueError):
        replace(settings, **change)
