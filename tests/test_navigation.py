import math
from pathlib import Path
import yaml
from brain.navigation import RealNavigation


class Map:
    def map(self, pixel):
        return [.22, 0] if pixel == [50, 100] else None


def test_real_parking_needs_matching_calibration_and_fresh_distinct_frames():
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'config/robot.yaml').read_text())
    nav = RealNavigation(cfg, Map())
    detection = dict(bbox_px=[0, 0, 100, 100], floor_m=[.22, 0])
    for frame in [1, 1, 2]:
        nav.observe(frame, 0, None)
        assert not nav.approach(detection)[0]
    nav.observe(3, 0, None)
    assert nav.approach(detection) == (True, (0, 0))
    nav.observe(4, .5, None)
    assert nav.approach(detection) == (False, (0, 0))
    detection['floor_m'] = [.3, 0]
    nav.observe(5, 0, None)
    assert nav.approach(detection) == (False, (0, 0))
    detection['bbox_px'] = [0, 0, 20, 20]
    assert nav.approach(detection) == (False, (0, 0))


def test_invalid_navigation_is_rejected_before_hardware():
    import pytest
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'config/robot.yaml').read_text())
    cfg['navigation']['settle_frames'] = 0
    with pytest.raises(ValueError):
        RealNavigation(cfg, Map())
