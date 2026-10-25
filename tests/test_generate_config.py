from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from base.firmware.generate_config import render

ROOT = Path(__file__).parents[1]
BASE = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())['base']


def test_current_config_matches_committed_pinmap():
    assert render(BASE) == (ROOT / 'base/firmware/include/pinmap.h').read_text()


@pytest.mark.parametrize('change, message', [
    ({'left_coil_pins': [8, 9, 10]}, 'exactly four'),
    ({'left_coil_pins': [8, 9, 10, 11, 12]}, 'exactly four'),
    ({'right_coil_pins': [2, 3, 4]}, 'exactly four'),
    ({'left_coil_pins': [8, 8, 10, 11]}, 'exactly four'),
    ({'right_coil_pins': [2, 3, 4, 4]}, 'exactly four'),
    ({'left_coil_pins': [8, 9, 10, 11], 'right_coil_pins': [2, 3, 4, 8]}, 'eight distinct'),
    ({'left_coil_pins': [8, 9, 10, 14]}, 'eight distinct'),
    ({'estop_sense_pin': 8}, 'E-stop'),
    ({'acceleration_mm_s2': 0}, 'positive'),
    ({'stop_acceleration_mm_s2': 0}, 'positive'),
    ({'acceleration_mm_s2': 1001, 'stop_acceleration_mm_s2': 2000}, '1000'),
    ({'stop_acceleration_mm_s2': 1001}, '1000'),
    ({'acceleration_mm_s2': 400, 'stop_acceleration_mm_s2': 300}, 'at least'),
    ({'max_speed_mm_s': 50, 'wheel_diameter_mm': 30}, 'half steps/s'),
    ({'max_speed_mm_s': 50, 'half_steps_per_rev': 8192}, 'half steps/s'),
])
def test_render_rejects_unsafe_config(change, message):
    config = deepcopy(BASE)
    config.update(change)
    with pytest.raises(ValueError, match=message):
        render(config)


def test_equal_accelerations_are_allowed():
    config = deepcopy(BASE)
    config.update(acceleration_mm_s2=300, stop_acceleration_mm_s2=300)
    render(config)
