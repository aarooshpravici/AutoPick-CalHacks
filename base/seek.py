"""Base-only seek: turn toward the detected item, creep up to it, STOP when close (no arm).

Uses the publisher's UDP detections (perception.publish must be running) in image
space, so it needs no floor calibration: it turns until the item's box is centred
in the RealSense colour image, drives forward, and stops when the box bottom nears
the image bottom or the aligned depth inside the box is short enough. Fake by
default (a simulated item that reacts to the commanded steps); --real needs an
explicit R4 port, human-verified wiring and a person's Enter. The arm is never
opened or checked: fold it at ready yourself before running.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from base.driver import BaseDriver, FakeSerial
from base.parking import Command

ROOT = Path(__file__).resolve().parents[1]
CAPS = dict(speed=15, distance_mm=3000, seconds=180)


class Seek:
    """Pure controller: detections in, wheel speeds out (mm/s, + forward, left wheel first)."""

    def __init__(self, width, height, *, speed=12, turn_speed=8, centred=.12, stop_depth_m=.35,
                 stop_row=.92, settle=3, lost_s=3.0, max_age_s=.3):
        self.width, self.height = width, height
        self.speed, self.turn_speed, self.centred = speed, turn_speed, centred
        self.stop_depth_m, self.stop_row, self.settle = stop_depth_m, stop_row, settle
        self.lost_s, self.max_age_s = lost_s, max_age_s
        self.close_frames, self.last_frame, self.seen_at = 0, None, None

    def update(self, detection, *, frame, age, now) -> Command:
        if self.seen_at is None:
            self.seen_at = now
        if detection is None or not 0 <= age <= self.max_age_s:
            self.close_frames = 0
            if now - self.seen_at > self.lost_s:
                return Command(reason="item not seen for %.0f s; stopped" % self.lost_s)
            return Command()                                 # hold still until it is seen again
        self.seen_at = now
        x1, _, x2, y2 = detection["bbox_px"]
        error = ((x1 + x2) / 2 - self.width / 2) / (self.width / 2)   # + means the item is to the right
        depth = detection.get("depth_m")
        close = y2 >= self.stop_row * self.height or (depth is not None and 0 < depth <= self.stop_depth_m)
        if close and abs(error) <= 2 * self.centred:
            if frame != self.last_frame:
                self.close_frames += 1
                self.last_frame = frame
            return Command(done=self.close_frames >= self.settle)
        self.close_frames = 0
        turn = self.turn_speed * max(-1, min(1, error / .5))
        if abs(error) > self.centred:
            return Command(turn, -turn)                     # rotate in place toward it
        forward = self.speed
        if depth is not None and depth > 0:                 # slow over the last 15 cm
            forward = max(5, min(self.speed, (depth - self.stop_depth_m) * 1000 / 15 * self.speed / 10))
        return Command(forward + turn * .5, forward - turn * .5)


class SimItem:
    """Fake mode: an item 0.8 m ahead and 15° right; its box follows the commanded steps."""

    def __init__(self, driver, config, label):
        self.driver, self.label, self.frame = driver, label, 0
        base = config["base"]
        self.mm_per_step = math.pi * base["wheel_diameter_mm"] / base["half_steps_per_rev"]
        self.track = base["track_width_mm"]
        self.width, self.height = config["realsense"]["colour"]
        self.x, self.y, self.heading = 0.0, 0.0, 0.0
        self.item = (800 * math.cos(math.radians(-15)), 800 * math.sin(math.radians(-15)))
        self.last = None

    def poll(self):
        status = self.driver.status(timeout=1)
        if self.last is not None:
            dl = (status.left_steps - self.last[0]) * self.mm_per_step
            dr = (status.right_steps - self.last[1]) * self.mm_per_step
            self.heading += (dr - dl) / self.track
            self.x += (dl + dr) / 2 * math.cos(self.heading)
            self.y += (dl + dr) / 2 * math.sin(self.heading)
        self.last = (status.left_steps, status.right_steps)
        dx, dy = self.item[0] - self.x, self.item[1] - self.y
        forward = math.cos(self.heading) * dx + math.sin(self.heading) * dy
        left = -math.sin(self.heading) * dx + math.cos(self.heading) * dy
        self.frame += 1
        bearing = math.atan2(left, forward)
        cx = self.width / 2 - bearing / math.radians(35) * self.width / 2
        y2 = self.height * min(1, .45 + 120 / max(forward, 1))
        items = [] if abs(bearing) > math.radians(35) or forward <= 0 else [dict(
            label=self.label, conf=.9, bbox_px=[cx - 60, y2 - 50, cx + 60, y2],
            centre_px=[cx, y2 - 25], floor_m=None, depth_m=math.hypot(forward, left) / 1000)]
        return dict(t=time.time(), frame=self.frame, items=items)

    def close(self):
        pass


def seek(driver, source, controller, label, *, distance_mm, seconds, real, config, log=print):
    try:
        driver.status(timeout=3)
        driver.synchronize()
        info = driver.verify_firmware(config["base"], require_ready=real)
        driver.set_travel_limit(distance_mm)
        started = time.monotonic()
        result = "timeout"
        while time.monotonic() - started < seconds:
            if driver.limit_reached:
                result = "travel limit reached"
                break
            message = source.poll()
            now = time.monotonic()
            if message is not None:
                items = [i for i in message["items"] if i["label"] == label]
                best = max(items, key=lambda i: i["conf"], default=None)
                age = max(0.0, time.time() - message["t"])
                command = controller.update(best, frame=message["frame"], age=age, now=now)
            else:
                command = controller.update(None, frame=None, age=math.inf, now=now)
            if command.done:
                result = "reached"
                break
            if command.reason:
                result = command.reason
                break
            driver.drive(command.left, command.right)
            time.sleep(.05)
    finally:
        driver.stop_and_wait()      # also on Ctrl-C, errors, timeout or an exhausted budget
    log(f"{label}: {result}; STOP acknowledged")
    return dict(result=result, firmware=info, stop_acknowledged=True,
                physical_direction_verified=False, arm_checked=False)


def main(argv=None):
    import yaml
    from brain.detections import UdpDetections
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.yaml")
    parser.add_argument("--item", default="sock")
    parser.add_argument("--speed", type=float, default=12, help="forward mm/s (cap 15)")
    parser.add_argument("--stop-depth-m", type=float, default=.35, help="stop when the box's depth is this close")
    parser.add_argument("--distance-mm", type=float, default=1500, help="firmware wheel-travel LIMIT (cap 3000)")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--port", help="explicit R4 port, required with --real")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    cap = min(CAPS["speed"], config["base"]["max_speed_mm_s"])
    for name, value, top in (("speed", args.speed, cap), ("distance-mm", args.distance_mm, CAPS["distance_mm"]),
                             ("seconds", args.seconds, CAPS["seconds"]), ("stop-depth-m", args.stop_depth_m, 1.5)):
        if not (math.isfinite(value) and 0 < value <= top):
            parser.error(f"--{name} must be in (0, {top:g}]")
    if args.item not in config["items"]:
        parser.error(f"--item must be one of robot.yaml items {config['items']}")
    width, height = config["realsense"]["colour"]
    controller = Seek(width, height, speed=args.speed, turn_speed=min(8, cap), stop_depth_m=args.stop_depth_m)
    if args.real:
        if config["base"].get("wiring_confirmed") is not True:
            parser.error("human-verified base.wiring_confirmed is required (check wiring and run base.jog first)")
        if not args.port or not args.port.startswith("/dev/") or "XXXX" in args.port:
            parser.error("explicit measured --port is required")
        normalize = lambda p: p.replace("/dev/tty.", "/dev/cu.")
        if normalize(args.port) in [normalize(config["arm"][key]) for key in ("follower_port", "leader_port")]:
            parser.error("refusing an arm port")
        if not sys.stdin.isatty():
            parser.error("a person must run --real in an interactive terminal")
        print(f"REAL WHEELS: seek the {args.item} at <= {args.speed:g} mm/s, stop at {args.stop_depth_m:g} m, "
              f"at most {args.distance_mm:g} mm wheel travel / {args.seconds:g} s.")
        print("perception.publish must be running. The ARM IS NOT CHECKED: fold it at ready yourself.")
        print("Clear the area and stay at the E-stop. Ctrl-C stops the wheels.")
        if input("Press Enter to start, or type anything to cancel: "):
            return 1
        import serial
        port = serial.Serial(args.port, baudrate=config["base"]["baud"], timeout=.05, write_timeout=.1, exclusive=True)
    else:
        port = FakeSerial(max_speed_mm_s=config["base"]["max_speed_mm_s"], config=config["base"],
                          wheel_diameter_mm=config["base"]["wheel_diameter_mm"],
                          steps_per_rev=config["base"]["half_steps_per_rev"])
    try:
        with BaseDriver(port, max_speed_mm_s=config["base"]["max_speed_mm_s"], max_lift_mm=1000, timeout=1) as driver:
            source = (UdpDetections(config["perception"]["detections_udp"]) if args.real
                      else SimItem(driver, config, args.item))
            try:
                result = seek(driver, source, controller, args.item, distance_mm=args.distance_mm,
                              seconds=args.seconds, real=args.real, config=config)
            finally:
                source.close()
            print(json.dumps(dict(result, simulated=not args.real), indent=2))
            return 0 if result["result"] == "reached" else 1
    except (Exception, KeyboardInterrupt) as error:
        print(f"Seek stopped: {error}. Use the physical E-stop if the wheels still move.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
