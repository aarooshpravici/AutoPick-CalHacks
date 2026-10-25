import subprocess
import shutil
from pathlib import Path
import pytest
import yaml
from base.firmware.generate_config import render

ROOT = Path(__file__).parents[1]


def test_actual_main_with_sensed_estop_and_zero_available_for_write(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('native compiler required')
    config = yaml.safe_load((ROOT / 'config/robot.yaml').read_text())['base']
    config.update(wiring_confirmed=True, estop_sense_pin=6)
    (tmp_path / 'pinmap.h').write_text(render(config))
    binary = tmp_path / 'main-test'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'tests/arduino'), '-I', str(tmp_path),
                    '-I', str(ROOT / 'base/firmware/include'),
                    str(ROOT / 'tests/r4_main_test.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=5)
