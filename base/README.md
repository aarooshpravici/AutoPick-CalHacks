# Base driver (B4)

The client implements the frozen line protocol in plan section 9.5. It belongs
in the Python 3.12 LeRobot environment (PyYAML and pyserial are needed for real
configuration and ports). It does not import LeRobot itself.

Run no-hardware tests:

```sh
python3.12 -m unittest discover -s tests -p 'test_base_driver.py' -v
```

Use `BaseDriver.from_config("config/robot.yaml", fake=True)` for simulation, or
inject `FakeSerial` into `BaseDriver`. The context manager sends STOP and closes
the port on exit. `status(timeout=1)` waits for the first status message;
`status()` returns the latest immutable status without waiting.

The mission loop must call `drive(left, right)` at 10 Hz. The driver never
renews a DRIVE command on its own. The fake firmware stops wheels at 300 ms
without a new DRIVE, and emits status every 100 ms. The default fake is wheels-only:
stretch commands raise `NoLiftError` (`ERR no lift`) without faulting the link.
Use `FakeSerial(has_lift=True, lift_speed_mm_s=100)` to test an interruptible
lift. Omitting its speed makes moves instantaneous. It is not a mechanical
model or a substitute for B3/L2 bench testing.

Wheel speeds are mm/s. Lift targets are mm above the home switch, limited by
the configured presets. Status includes wheel step counts, lift mm, tray grams,
E-stop and a monotonic reception time. Nonzero driving requires fresh status,
E-stop released, and tray at zero with no lift command in progress. Homing is
required by the fake firmware before lifting. Every lift command first sends
STOP to the wheels. The real firmware must enforce
the same physical interlocks independently.

Replies and asynchronous STAT messages are separated by a background reader.
Requests serialize because there are no request IDs in the protocol. STOP
bypasses request waits. Serial faults, firmware ERR (except `ERR no lift`), malformed status and command
timeouts require reconnecting; this prevents late replies from being attributed
to a new command. Callers can still attempt STOP after a fault.

**Approved STOP contract (REQ-codex-2):** STOP cancels an active lift move, holds
the lift enabled at its current height, stops wheels and replies `OK STOP`.
The waiting lift caller raises `MotionStopped` after draining that acknowledgment;
it does not wait for the abandoned target or poison the link. Driving stays
blocked until a new home/lower operation completes. `stop()` sends without waiting
for its reply, but serial writes can block up to the configured write timeout;
M1 must dispatch it off the UI event loop. Physical E-stop cuts power and cannot
promise lift holding. Bench STOP/hold timing remains unverified.
Do not open real ports or move hardware without SW2 confirming the area is clear.
