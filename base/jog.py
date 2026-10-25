"""One bounded wheel-direction test. Fake by default; --real requires a person's Enter."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from base.driver import BaseDriver, FakeSerial

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {'forward': (1, 1), 'back': (-1, -1), 'left': (-1, 1), 'right': (1, -1)}


def validate(config, direction, speed, duration, distance_mm):
    cap = config['base']['max_speed_mm_s']
    if direction not in DIRECTIONS:
        raise ValueError('invalid direction')
    for value in (speed, duration, distance_mm, cap):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('speed, duration, distance and configured cap must be positive finite numbers')
    if speed > min(10, cap) or duration > 2 or distance_mm > 20:
        raise ValueError('bring-up caps: speed <= min(10, configured cap) mm/s, duration <= 2 s, travel <= 20 mm')


def jog(driver, config, direction, speed, duration, distance_mm):
    validate(config, direction, speed, duration, distance_mm)
    try:
        driver.status(timeout=3)
        driver.synchronize()
        info = driver.verify_firmware(config['base'], require_ready=not isinstance(driver.port, FakeSerial))
        before = driver.fresh_status(after=time.monotonic())
        driver.set_travel_limit(distance_mm)
        started = time.monotonic()
        left, right = DIRECTIONS[direction]
        while time.monotonic() - started < duration and not driver.limit_reached:
            driver.drive(left * speed, right * speed)
            time.sleep(min(.05, max(0, duration - (time.monotonic() - started))))
    finally:
        driver.stop_and_wait()  # Also on Ctrl-C, timeout, error or an exhausted budget.
    after = driver.fresh_status(after=time.monotonic())
    steps = dict(left=after.left_steps-before.left_steps, right=after.right_steps-before.right_steps)
    mm_per_step = math.pi * config['base']['wheel_diameter_mm'] / config['base']['half_steps_per_rev']
    return dict(direction=direction, speed_mm_s=speed, max_seconds=duration,
                firmware=info, commanded_steps=steps,
                estimated_wheel_mm={key: value * mm_per_step for key, value in steps.items()},
                physical_direction_verified=False, physical_estop_verified=False,
                max_wheel_travel_mm=distance_mm, limit_reached=driver.limit_reached,
                stop_acknowledged=True, note='Observe actual direction; this does not certify traction or E-stop.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.yaml')
    parser.add_argument('--direction', choices=DIRECTIONS, required=True)
    parser.add_argument('--speed', type=float, default=5)
    parser.add_argument('--seconds', type=float, default=1)
    parser.add_argument('--distance-mm', type=float, default=10)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--port', help='explicit R4 port, required with --real')
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text())
    validate(config, args.direction, args.speed, args.seconds, args.distance_mm)
    if args.real:
        if config['base'].get('wiring_confirmed') is not True:
            parser.error('human-verified wiring_confirmed is required')
        if not args.port or not args.port.startswith('/dev/') or 'XXXX' in args.port:
            parser.error('explicit measured --port is required')
        normalize = lambda p: p.replace('/dev/tty.', '/dev/cu.')
        if normalize(args.port) in [normalize(config['arm'][key]) for key in ('follower_port', 'leader_port')]:
            parser.error('refusing an arm port')
        if not sys.stdin.isatty():
            parser.error('a person must run --real in an interactive terminal')
        print(f'REAL WHEELS: {args.direction}, {args.speed:g} mm/s, at most {args.seconds:g} s / {args.distance_mm:g} mm wheel travel.')
        print('Close the robot server/serial monitor. Raise wheels for first tests. Clear the area; stay at the E-stop.')
        if input('Press Enter to run this one test, or type anything to cancel: '):
            return 1
        import serial
        port = serial.Serial(args.port, baudrate=config['base']['baud'], timeout=.05, write_timeout=.1, exclusive=True)
    else:
        port = FakeSerial(max_speed_mm_s=config['base']['max_speed_mm_s'], config=config['base'],
                          wheel_diameter_mm=config['base']['wheel_diameter_mm'],
                          steps_per_rev=config['base']['half_steps_per_rev'])
    try:
        with BaseDriver(port, max_speed_mm_s=config['base']['max_speed_mm_s'], max_lift_mm=1000, timeout=1) as driver:
            result = jog(driver, config, args.direction, args.speed, args.seconds, args.distance_mm)
            print(json.dumps(dict(result, simulated=not args.real), indent=2))
    except (Exception, KeyboardInterrupt) as error:
        print(f'Test failed; use physical E-stop if motion persists: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
