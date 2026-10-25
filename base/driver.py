"""Section 9.5 serial client. Call drive at 10 Hz; never auto-renew motion."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

WATCHDOG_S = 0.3
STATUS_PERIOD_S = 0.1


class MotionStopped(RuntimeError):
    """An outstanding lift movement was abandoned by STOP."""


class NoLiftError(RuntimeError):
    """The connected wheels-only firmware has no lift/load cell."""


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


@dataclass(frozen=True)
class Status:
    left_steps: int
    right_steps: int
    lift_mm: float
    tray_g: float
    estop: bool
    received_at: float


class BaseDriver:
    def __init__(self, serial_port, *, max_speed_mm_s, max_lift_mm, timeout=5.0):
        self.port = serial_port
        self.max_speed = finite(max_speed_mm_s)
        self.max_lift = finite(max_lift_mm)
        self.timeout = finite(timeout)
        if min(self.max_speed, self.max_lift, self.timeout) <= 0:
            raise ValueError("limits and timeout must be positive")
        self._condition = threading.Condition()
        self._request_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._replies = deque(maxlen=128)
        self._status = None
        self.limit_reached = False
        self._fault = None
        self._lift_blocked = False
        self._stop_generation = 0
        self._stop_acked = 0
        self._closed = threading.Event()
        self._reader = threading.Thread(target=self._read, daemon=True, name="base-status")
        self._reader.start()

    @classmethod
    def from_config(cls, path=Path("config/robot.yaml"), *, fake=False, timeout=5.0):
        import yaml
        with open(path) as stream:
            cfg = yaml.safe_load(stream)
        base, lift = cfg["base"], cfg["lift"]
        max_lift = max(lift["seated_hand_mm"], lift["standing_hand_mm"])
        if fake:
            port = FakeSerial(max_speed_mm_s=base["max_speed_mm_s"], max_lift_mm=max_lift,
                              wheel_diameter_mm=base["wheel_diameter_mm"],
                              steps_per_rev=base.get("half_steps_per_rev", 1600), config=base)
        else:
            import serial
            port = serial.Serial(base["serial_port"], baudrate=base["baud"],
                                 timeout=0.05, write_timeout=0.1, exclusive=True)
        return cls(port, max_speed_mm_s=base["max_speed_mm_s"], max_lift_mm=max_lift,
                   timeout=timeout)

    def _read(self):
        buffer = b""
        try:
            while not self._closed.is_set():
                buffer += self.port.readline()
                if len(buffer) > 8192:
                    raise RuntimeError("oversized serial input")
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    line = raw.decode("ascii").strip()
                    if not line:
                        continue
                    with self._condition:
                        if line.startswith("STAT "):
                            fields = line.split()
                            if len(fields) != 6 or fields[5] not in ("0", "1"):
                                raise RuntimeError("malformed STAT")
                            self._status = Status(int(fields[1]), int(fields[2]),
                                                  finite(fields[3]), finite(fields[4]),
                                                  fields[5] == "1", time.monotonic())
                        else:
                            if line == "LIMIT REACHED":
                                self.limit_reached = True
                            if line == "OK LIMIT":
                                self.limit_reached = False
                            if line == "OK STOP":
                                self._stop_acked += 1
                            if line.startswith("ERR ") and line != "ERR no lift":
                                self._fault = RuntimeError(line)
                            self._replies.append(line)
                        self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._fault = error
                self._condition.notify_all()

    def _send(self, command):
        try:
            with self._write_lock:
                if self._closed.is_set():
                    raise RuntimeError("serial driver closed")
                data = (command + "\n").encode("ascii")
                if self.port.write(data) != len(data):
                    raise OSError("partial serial write")
        except Exception as error:
            with self._condition:
                self._fault = error
                self._condition.notify_all()
            raise

    def _healthy(self):
        if self._closed.is_set():
            raise RuntimeError("serial driver closed")
        if self._fault is not None:
            raise RuntimeError(f"serial link failed: {self._fault}; reconnect before continuing") from self._fault

    def _request(self, command, matches):
        # Single outstanding request because the frozen protocol has no request IDs.
        # STOP bypasses this lock and can interrupt a long lift wait.
        with self._request_lock:
            with self._condition:
                self._healthy()
                self._replies.clear()
                generation = self._stop_generation
                self._send(command)
                deadline = time.monotonic() + self.timeout
                while True:
                    self._healthy()
                    cancelled = command.startswith("LIFT_") and generation != self._stop_generation
                    if cancelled and self._stop_acked >= self._stop_generation:
                        raise MotionStopped("lift stopped and holding")
                    while self._replies:
                        reply = self._replies.popleft()
                        if cancelled:
                            # Drain through the STOP acknowledgment before another lift
                            # request can start: late completion replies cannot leak.
                            continue
                        if reply == "ERR no lift":
                            raise NoLiftError(reply)
                        if reply.startswith("ERR "):
                            raise RuntimeError(reply)
                        if matches(reply):
                            return reply
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._fault = TimeoutError(command)
                        raise self._fault
                    self._condition.wait(remaining)

    def status(self, *, timeout=0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._healthy()
                if self._status is not None:
                    return self._status
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no STAT received")
                self._condition.wait(remaining)

    def fresh_status(self, *, after, timeout=.4):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._healthy()
                if self._status is not None and self._status.received_at > after:
                    return self._status
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no fresh STAT received")
                self._condition.wait(remaining)

    def synchronize(self, timeout=.4):
        """Cancel partial RX text and halt, then require a fresh STOP acknowledgment."""
        self._send("\x18")
        self.stop_and_wait(timeout=timeout)

    def firmware_info(self):
        reply = self._request("INFO", lambda reply: reply.startswith("INFO "))
        try:
            tag, family, version, *tokens = reply.split()
            fields = dict(token.split("=", 1) for token in tokens)
            if family != "floorpicker-r4" or version != "2" or len(fields) != len(tokens):
                raise ValueError("unexpected firmware identity")
            if fields['limit'] != '1' or fields['hold'] != '1' or fields['watchdog_ms'] != '300':
                raise ValueError("required LIMIT/hold/watchdog capabilities missing")
            if fields['wiring'] not in ('0','1') or fields['estop_sense'] not in ('0','1'):
                raise ValueError("invalid firmware flags")
            if len(fields['config']) != 16 or any(c not in '0123456789abcdef' for c in fields['config']):
                raise ValueError("invalid firmware configuration fingerprint")
            speed = finite(fields['max_mm_s'])
            if not 0 < speed <= 50:
                raise ValueError("invalid firmware speed cap")
            return dict(family=family, version=int(version), limit=True, hold=True,
                        watchdog_ms=300, wiring=fields['wiring']=='1',
                        estop_sensed=fields['estop_sense']=='1', max_speed_mm_s=speed,
                        config_id=fields['config'])
        except (ValueError, KeyError) as error:
            raise RuntimeError(f"incompatible R4 INFO: {error}") from error

    def verify_firmware(self, base, *, require_ready=True):
        from base.firmware.generate_config import config_id
        info = self.firmware_info()
        if info['config_id'] != config_id(base):
            raise RuntimeError("R4 firmware configuration differs from robot.yaml; regenerate/review/reflash together")
        if not math.isclose(info['max_speed_mm_s'], float(base['max_speed_mm_s']), rel_tol=1e-6, abs_tol=.0005):
            raise RuntimeError("R4 speed cap differs from robot.yaml")
        if require_ready and not info['wiring']:
            raise RuntimeError("R4 motor outputs disabled: wiring has not been confirmed in installed firmware")
        return info

    def set_travel_limit(self, mm):
        mm = finite(mm)
        if not 0 < mm <= 6096:
            raise ValueError("travel limit must be greater than 0 and at most 6096 mm")
        self._request(f"LIMIT {mm:.6f}", lambda reply: reply == "OK LIMIT")

    def ping(self):
        self._request("PING", lambda reply: reply == "PONG")

    def drive(self, left, right):
        left, right = finite(left), finite(right)
        if max(abs(left), abs(right)) > self.max_speed:
            raise ValueError("wheel speed exceeds configured maximum")
        if left or right:
            status = self.status()
            if time.monotonic() - status.received_at > WATCHDOG_S:
                raise RuntimeError("stale base status")
            if status.estop or status.lift_mm != 0 or self._lift_blocked:
                raise RuntimeError("drive requires E-stop released and tray homed")
        self._send(f"DRIVE {left:g} {right:g}")

    def stop(self):
        # Bypasses request waits; firmware holds the lift motor enabled.
        with self._condition:
            self._stop_generation += 1
            self._send("STOP")
            self._condition.notify_all()

    def stop_and_wait(self, timeout=.4):
        """Wait for firmware's stopped acknowledgment without blocking UI callers."""
        deadline = time.monotonic() + timeout
        with self._condition:
            self.stop()
            ticket = self._stop_generation
            while self._stop_acked < ticket:
                self._healthy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fault = TimeoutError("STOP acknowledgment missing")
                    raise self._fault
                self._condition.wait(remaining)

    def lift_home(self):
        self._lift_blocked = True
        self.stop()
        try:
            self._request("LIFT_HOME", lambda reply: reply == "LIFT_HOMED")
        except NoLiftError:
            self._lift_blocked = False
            raise
        self._lift_blocked = False

    def lift_to(self, mm):
        mm = finite(mm)
        if not 0 <= mm <= self.max_lift:
            raise ValueError("lift height outside configured presets")
        target = f"{mm:g}"
        self._lift_blocked = True
        self.stop()
        try:
            self._request(f"LIFT_TO {target}",
                          lambda reply: reply.startswith("LIFT_AT ") and finite(reply[8:]) == float(target))
        except NoLiftError:
            self._lift_blocked = False
            raise
        self._lift_blocked = mm != 0

    def tray_grams(self):
        return finite(self._request("TRAY", lambda reply: reply.startswith("TRAY "))[5:])

    def tare(self):
        self._request("TARE", lambda reply: reply == "OK TARE")

    def close(self):
        if not self._closed.is_set():
            try:
                self.stop()
            finally:
                self._closed.set()
                self.port.close()
                with self._condition:
                    self._condition.notify_all()
                self._reader.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeSerial:
    """In-memory wheels-only firmware, with an optional simulated stretch lift.

    Use an injected monotonic clock for deterministic watchdog tests. This model
    verifies protocol behavior, not lift mechanics or electrical safety.
    """
    def __init__(self, *, max_speed_mm_s=200, max_lift_mm=1000, wheel_diameter_mm=72,
                 clock=time.monotonic, has_lift=False, lift_speed_mm_s=None, steps_per_rev=1600, config=None):
        self.clock = clock
        self.config = config
        self.max_speed = max_speed_mm_s
        self.max_lift = max_lift_mm
        self.has_lift = has_lift
        self.lift_speed = None if lift_speed_mm_s is None else finite(lift_speed_mm_s)
        if self.lift_speed is not None and self.lift_speed <= 0:
            raise ValueError("lift speed must be positive")
        self.lift_target = None
        self.lift_enabled = False
        self._homing = False
        self.steps_per_mm = steps_per_rev / (math.pi * wheel_diameter_mm)
        self.left = self.right = self.lift_mm = self.weight = 0.0
        self.left_steps = self.right_steps = 0.0
        self.travel_budget = None
        self.travel_used = 0.0
        self.limit_reached = False
        self.estop = False
        self.homed = False
        self.closed = False
        self.last_drive = self.last_update = clock()
        self.last_status = clock() - STATUS_PERIOD_S
        self._lines = deque()
        self._lock = threading.Lock()
        self.commands = []

    def _tick(self):
        now = self.clock()
        elapsed = max(0, now - self.last_update)
        moving_dt = max(0, min(now, self.last_drive + WATCHDOG_S) - self.last_update)
        speed = (abs(self.left) + abs(self.right)) / 2
        if self.travel_budget is not None and speed:
            moving_dt = min(moving_dt, max(0, self.travel_budget - self.travel_used) / speed)
            self.travel_used += speed * moving_dt
        self.left_steps += self.left * moving_dt * self.steps_per_mm
        self.right_steps += self.right * moving_dt * self.steps_per_mm
        if self.travel_budget is not None and self.travel_used >= self.travel_budget:
            if not self.limit_reached:
                self._lines.append(b"LIMIT REACHED\n")
            self.limit_reached = True
            self.left = self.right = 0
        self.last_update = now
        if now - self.last_drive >= WATCHDOG_S or self.estop:
            self.left = self.right = 0.0
        if self.estop:
            self.lift_target = None
            self.lift_enabled = False
            self.homed = False
        if self.lift_target is not None:
            delta = self.lift_target - self.lift_mm
            step = abs(delta) if self.lift_speed is None else self.lift_speed * elapsed
            self.lift_mm += max(-step, min(step, delta))
            if self.lift_mm == self.lift_target:
                if self._homing:
                    self.homed = True
                reply = "LIFT_HOMED" if self._homing else f"LIFT_AT {self.lift_mm:g}"
                self._lines.append((reply + "\n").encode())
                self.lift_target = None

    def _move_lift(self, height, *, homing=False):
        self.lift_enabled = True
        self._homing = homing
        self.lift_target = height
        self._tick()

    def _status(self):
        return f"STAT {int(self.left_steps)} {int(self.right_steps)} {self.lift_mm:g} {self.weight:g} {int(self.estop)}"

    def write(self, data):
        with self._lock:
            if self.closed:
                raise OSError("fake serial closed")
            self._tick()
            command = data.decode("ascii").strip()
            self.commands.append(command)
            parts = command.split()
            reply = None
            try:
                if parts == ["\x18"]:
                    self.left = self.right = 0
                elif parts == ["INFO"]:
                    if self.config is None:
                        reply = "ERR INFO unavailable in this fake configuration"
                    else:
                        from base.firmware.generate_config import config_id
                        cfg = self.config
                        reply = (f"INFO floorpicker-r4 2 limit=1 hold=1 watchdog_ms=300 "
                                 f"wiring={int(cfg['wiring_confirmed'])} estop_sense={int(cfg['estop_sense_pin'] >= 0)} "
                                 f"max_mm_s={cfg['max_speed_mm_s']:g} config={config_id(cfg)}")
                elif parts == ["PING"]:
                    reply = "PONG"
                elif len(parts) == 2 and parts[0] == "LIMIT":
                    mm = finite(parts[1])
                    if not 0 < mm <= 6096 or self.left or self.right:
                        raise ValueError("invalid travel limit")
                    self.travel_budget, self.travel_used, self.limit_reached = mm, 0, False
                    reply = "OK LIMIT"
                elif parts == ["STOP"]:
                    self.left = self.right = 0
                    self.lift_target = None  # Keep enable and current height: hold.
                    reply = "OK STOP"
                elif parts and parts[0] in ("TRAY", "TARE", "LIFT_HOME", "LIFT_TO") and not self.has_lift:
                    reply = "ERR no lift"
                elif parts == ["TRAY"]:
                    reply = f"TRAY {self.weight:g}"
                elif parts == ["TARE"]:
                    self.weight = 0
                    reply = "OK TARE"
                elif parts == ["LIFT_HOME"] and not self.estop:
                    if self.left or self.right:
                        raise ValueError("stop wheels before homing")
                    self._move_lift(0, homing=True)
                elif len(parts) == 2 and parts[0] == "LIFT_TO" and not self.estop:
                    height = finite(parts[1])
                    if not self.homed or not 0 <= height <= self.max_lift or self.left or self.right:
                        raise ValueError("home lift first or invalid height")
                    self._move_lift(height)
                elif len(parts) == 3 and parts[0] == "DRIVE" and not self.estop:
                    left, right = finite(parts[1]), finite(parts[2])
                    if max(abs(left), abs(right)) > self.max_speed or self.lift_mm != 0 or self.lift_target is not None:
                        raise ValueError("unsafe drive")
                    if self.limit_reached:
                        reply = "LIMIT REACHED"
                    else:
                        self.left, self.right, self.last_drive = left, right, self.clock()
                else:
                    raise ValueError("unknown or disabled command")
            except ValueError as error:
                reply = f"ERR {error}"
            if reply is not None:
                self._lines.append((reply + "\n").encode("ascii"))
            # Also report changed status, keeping test consumers current.
            if parts and parts[0] in ("LIFT_HOME", "LIFT_TO", "TARE"):
                self._lines.append((self._status() + "\n").encode("ascii"))
        return len(data)

    def readline(self):
        with self._lock:
            if self.closed:
                return b""
            self._tick()
            if self._lines:
                return self._lines.popleft()
            if self.clock() - self.last_status >= STATUS_PERIOD_S:
                self.last_status = self.clock()
                return (self._status() + "\n").encode("ascii")
        time.sleep(0.005)
        return b""

    def close(self):
        with self._lock:
            self.closed = True
