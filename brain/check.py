"""Read-only demo preflight. Never opens a camera, serial port or servo bus."""
import argparse
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def inspect(config_path, role, *, root=ROOT):
    checks = []
    def run(name, fn):
        try:
            detail = fn()
            checks.append(dict(check=name, ok=True, detail=detail or 'OK'))
        except Exception as error:
            checks.append(dict(check=name, ok=False, detail=str(error)))

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    import yaml
    try:
        config = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(config, dict):
            raise ValueError('robot.yaml must be a mapping')
    except Exception as error:
        return [dict(check='configuration', ok=False, detail=str(error))]
    expected = (3, 10) if role == 'perception' else (3, 12)
    run('Python version', lambda: require(sys.version_info[:2] == expected,
        f'Use Python {expected[0]}.{expected[1]} for {role}; found {sys.version.split()[0]}'))
    modules = ('numpy', 'cv2', 'pyrealsense2', 'ultralytics') if role == 'perception' else ('serial', 'fastapi', 'uvicorn', 'lerobot', 'numpy', 'cv2')
    for module in modules:
        run(f'module {module}', lambda module=module: require(importlib.util.find_spec(module) is not None, f'{module} missing in this interpreter'))
    if role == 'robot':
        def lerobot_version():
            found = importlib.metadata.version('lerobot')
            require(found == '0.6.2', f'LeRobot 0.6.2 required; found {found}')
            return found
        run('LeRobot version', lerobot_version)
        from brain.mission import grasp_settings, handoff_timeout
        def poses():
            from arm.poses import load
            grasp, _ = grasp_settings(config)
            handoff_timeout(config)
            load(root / 'config/poses.json', mode='suction' if grasp == 'suction' else 'hand')
            return f'{grasp} pose schema valid; on-base teaching still needs human verification'
        run('grasp and poses', poses)
        def arm_config():
            from arm.pick import load_arm_config
            cfg = load_arm_config(config_path, require=('follower_port', 'follower_id', 'leader_port', 'leader_id'))
            require('wrist_roll' in cfg.get('frozen_joints', []), 'faulty wrist_roll must remain frozen')
            require(cfg['max_relative_target_deg'] <= 10, 'relative arm step cap must be <=10 degrees')
        run('arm configuration and frozen wrist', arm_config)
        def wiring():
            from base.firmware.generate_config import render
            base = config['base']
            require(base.get('wiring_confirmed') is True, 'wiring_confirmed is not true; human verification pending')
            require(base.get('drive_calibrated') is True, 'drive_calibrated is not true; loaded checks pending')
            require((root / 'base/firmware/include/pinmap.h').read_text() == render(base), 'firmware pinmap differs from robot.yaml; regenerate and review before flashing')
            return 'configured flags/header agree; does not inspect installed firmware or physical E-stop'
        run('base configuration', wiring)
        def port():
            value = config['base']['serial_port']
            require(isinstance(value, str) and value.startswith('/dev/') and 'XXXX' not in value, 'set actual R4 serial port')
            normalize = lambda p: p.replace('/dev/tty.', '/dev/cu.')
            require(normalize(value) not in [normalize(config['arm'][key]) for key in ('follower_port','leader_port')], 'R4 port must not be an arm port')
            require(Path(value).exists(), 'configured R4 port does not exist on this Mac')
            return 'path exists; not opened or tested'
        run('R4 port', port)
    def floor():
        from perception.floor import FloorMap
        from brain.navigation import RealNavigation
        mapping = FloorMap.load(root / 'config/floor_homography.json', colour_size=config['realsense']['colour'], flip=config['realsense']['flip'])
        require(math.dist(mapping.parking(), config['perception']['parking_spot_floor_m']) <= 1e-6, 'calibrated parking target differs from robot.yaml')
        RealNavigation(config, mapping)
        return 'stored fit/check evidence and parking target pass; mount must remain unchanged'
    run('floor and pickup target', floor)
    def weights():
        value = Path(config['perception']['detector_weights'])
        path = value if value.is_absolute() else root / value
        require(path.is_file() and path.stat().st_size > 0, f'missing or empty detector weights: {path}')
        return 'file exists; class names and live accuracy must be verified by publisher/live checks'
    run('phone checkpoint', weights)
    run('demo object', lambda: require(config.get('items') in (['phone'], ['phone', 'sock']), 'demo integration supports phone and sock only'))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.yaml')
    parser.add_argument('--role', choices=('robot','perception'), required=True)
    args = parser.parse_args(argv)
    try:
        checks = inspect(args.config, args.role)
    except Exception as error:
        checks = [dict(check='preflight', ok=False, detail=str(error))]
    print(json.dumps(dict(role=args.role, hardware_opened=False, checks=checks), indent=2))
    return 0 if checks and all(row['ok'] for row in checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
