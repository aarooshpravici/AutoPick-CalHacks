"""Check R4 identity/configuration and PING/STAT/STOP; never sends DRIVE or LIMIT."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from base.driver import BaseDriver

ROOT = Path(__file__).resolve().parents[1]


def check(driver, base=None):
    driver.status(timeout=3)  # Allow the controller to boot after opening USB.
    start = time.monotonic()
    driver.synchronize(timeout=.5)
    elapsed = time.monotonic() - start
    driver.ping()
    info = driver.firmware_info() if base is None else driver.verify_firmware(base, require_ready=False)
    status = driver.status()
    if time.monotonic() - status.received_at > .3:
        raise RuntimeError('R4 status stopped arriving')
    if status.estop:
        raise RuntimeError('R4 reports E-stop active')
    return dict(protocol='9.5 / firmware 2', firmware=info,
                warnings=[] if info['estop_sensed'] else ['Physical E-stop is NOT sensed; STAT estop=0 is not a safety confirmation'],
                pong=True, stop_acknowledged=True,
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
    try:
        port = serial.Serial(args.port, baudrate=config['baud'], timeout=.05, write_timeout=.1, exclusive=True)
        with BaseDriver(port, max_speed_mm_s=config['max_speed_mm_s'], max_lift_mm=1000, timeout=2) as driver:
            print(json.dumps(check(driver, config), indent=2))
    except Exception as error:
        print(json.dumps(dict(ok=False, port=args.port, error=str(error),
                             next_step="Check the selected R4 port/cable, close other serial apps, and verify matching firmware/configuration. No DRIVE was sent."), indent=2))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
