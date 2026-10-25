"""Check the R4 USB serial link with PING/STAT/STOP only; never sends DRIVE."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from base.driver import BaseDriver

ROOT = Path(__file__).resolve().parents[1]


def check(driver):
    driver.status(timeout=3)  # Allow the controller to boot after opening USB.
    driver.ping()
    start = time.monotonic()
    driver.stop_and_wait(timeout=.5)
    elapsed = time.monotonic() - start
    status = driver.status()
    if time.monotonic() - status.received_at > .3:
        raise RuntimeError('R4 status stopped arriving')
    if status.estop:
        raise RuntimeError('R4 reports E-stop active')
    return dict(protocol='9.5', pong=True, stop_acknowledged=True,
                stop_round_trip_ms=round(elapsed * 1000, 1), status=asdict(status),
                note='Link check only; this does not measure physical stopping or verify motor wiring.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='list local serial ports without opening them')
    parser.add_argument('--port', help='R4 USB serial port on the Mac physically connected to it')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.yaml')
    args = parser.parse_args(argv)
    if args.list:
        from serial.tools.list_ports import comports
        print(json.dumps([dict(port=p.device, description=p.description,
                               manufacturer=p.manufacturer) for p in comports()], indent=2))
        return 0
    if not args.port:
        parser.error('use --list or explicitly select --port')
    import serial
    import yaml
    config = yaml.safe_load(args.config.read_text())['base']
    port = serial.Serial(args.port, baudrate=config['baud'], timeout=.05, write_timeout=.1)
    with BaseDriver(port, max_speed_mm_s=config['max_speed_mm_s'], max_lift_mm=1000, timeout=2) as driver:
        print(json.dumps(check(driver), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
