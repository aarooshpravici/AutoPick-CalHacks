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
        self._fault = None
        self._lift_blocked = False
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
                              wheel_diameter_mm=base["wheel_diameter_mm"])
        else:
            import serial
            port = serial.Serial(base["serial_port"], baudrate=base["baud"],
                                 timeout=0.05, write_timeout=0.1)
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
                            if line.startswith("ERR "):
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
            raise RuntimeError("serial link failed; reconnect before continuing") from self._fault

    def _request(self, command, matches):
        # Single outstanding request because the frozen protocol has no request IDs.
        # STOP bypasses this lock and can interrupt a long lift wait.
        with self._request_lock:
            with self._condition:
                self._healthy()
                self._replies.clear()
                self._send(command)
                deadline = time.monotonic() + self.timeout
                while True:
                    self._healthy()
                    while self._replies:
                        reply = self._replies.popleft()
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
        # Frozen protocol: STOP stops wheels only, not the lift.
        self._send("STOP")

    def lift_home(self):
        self._lift_blocked = True
        self.stop()
        self._request("LIFT_HOME", lambda reply: reply == "LIFT_HOMED")
        self._lift_blocked = False

    def lift_to(self, mm):
        mm = finite(mm)
        if not 0 <= mm <= self.max_lift:
            raise ValueError("lift height outside configured presets")
        target = f"{mm:g}"
        self._lift_blocked = True
        self.stop()
        self._request(f"LIFT_TO {target}",
                      lambda reply: reply.startswith("LIFT_AT ") and finite(reply[8:]) == float(target))
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
    """In-memory firmware model. Lift completes instantly; drive watchdog is real.

    Use an injected monotonic clock for deterministic watchdog tests. This model
    verifies protocol behavior, not lift mechanics or electrical safety.
    """
    def __init__(self, *, max_speed_mm_s=200, max_lift_mm=1000, wheel_diameter_mm=72,
                 clock=time.monotonic):
        self.clock = clock
        self.max_speed = max_speed_mm_s
        self.max_lift = max_lift_mm
        self.steps_per_mm = 1600 / (math.pi * wheel_diameter_mm)
        self.left = self.right = self.lift_mm = self.weight = 0.0
        self.left_steps = self.right_steps = 0.0
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
        moving_dt = max(0, min(now, self.last_drive + WATCHDOG_S) - self.last_update)
        self.left_steps += self.left * moving_dt * self.steps_per_mm
        self.right_steps += self.right * moving_dt * self.steps_per_mm
        self.last_update = now
        if now - self.last_drive >= WATCHDOG_S or self.estop:
            self.left = self.right = 0.0

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
                if parts == ["PING"]:
                    reply = "PONG"
                elif parts == ["STOP"]:
                    self.left = self.right = 0
                    reply = "OK STOP"
                elif parts == ["TRAY"]:
                    reply = f"TRAY {self.weight:g}"
                elif parts == ["TARE"]:
                    self.weight = 0
                    reply = "OK TARE"
                elif parts == ["LIFT_HOME"] and not self.estop:
                    self.lift_mm, self.homed = 0, True
                    reply = "LIFT_HOMED"
                elif len(parts) == 2 and parts[0] == "LIFT_TO" and not self.estop:
                    height = finite(parts[1])
                    if not self.homed or not 0 <= height <= self.max_lift or self.left or self.right:
                        raise ValueError("home lift first or invalid height")
                    self.lift_mm = height
                    reply = f"LIFT_AT {height:g}"
                elif len(parts) == 3 and parts[0] == "DRIVE" and not self.estop:
                    left, right = finite(parts[1]), finite(parts[2])
                    if max(abs(left), abs(right)) > self.max_speed or self.lift_mm != 0:
                        raise ValueError("unsafe drive")
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
