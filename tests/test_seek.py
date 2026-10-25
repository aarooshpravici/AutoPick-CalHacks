"""base.seek: image-space turn-and-approach controller. No hardware."""
import pytest

from base.seek import Seek, main

W, H = 1280, 720


def box(cx, y2, depth=None):
    return dict(label="sock", conf=.9, bbox_px=[cx - 50, y2 - 40, cx + 50, y2], depth_m=depth)


def test_turns_toward_an_item_on_the_right_in_place():
    command = Seek(W, H).update(box(1100, 400), frame=1, age=0, now=0)
    assert command.left > 0 > command.right and not command.done


def test_turns_toward_an_item_on_the_left_in_place():
    command = Seek(W, H).update(box(150, 400), frame=1, age=0, now=0)
    assert command.right > 0 > command.left


def test_drives_forward_when_centred_and_slows_near_the_stop_depth():
    seek = Seek(W, H, speed=12)
    far = seek.update(box(640, 400, depth=1.0), frame=1, age=0, now=0)
    near = seek.update(box(640, 400, depth=.4), frame=2, age=0, now=.1)
    assert far.left == pytest.approx(12) and far.right == pytest.approx(12)
    assert 0 < near.left < far.left


def test_stops_after_three_distinct_close_frames():
    seek = Seek(W, H)
    results = [seek.update(box(640, 700, depth=.3), frame=f, age=0, now=f * .1) for f in (1, 1, 2, 3)]
    assert [r.done for r in results] == [False, False, False, True]
    assert all(r.left == r.right == 0 for r in results)


def test_box_bottom_at_image_bottom_counts_as_close_without_depth():
    seek = Seek(W, H)
    for frame in (1, 2):
        seek.update(box(640, 700), frame=frame, age=0, now=frame)
    assert seek.update(box(640, 700), frame=3, age=0, now=3).done


def test_stale_or_missing_detection_holds_still_then_gives_up():
    seek = Seek(W, H, lost_s=3)
    assert seek.update(box(640, 400), frame=1, age=0, now=0).left != 0
    held = seek.update(box(640, 400), frame=2, age=.5, now=1)
    assert held.left == held.right == 0 and not held.reason
    lost = seek.update(None, frame=None, age=float("inf"), now=4.5)
    assert lost.left == lost.right == 0 and "not seen" in lost.reason


def test_real_refuses_without_verified_wiring(capsys):
    with pytest.raises(SystemExit):
        main(["--real", "--port", "/dev/cu.usbmodemX"])
    assert "wiring_confirmed" in capsys.readouterr().err
