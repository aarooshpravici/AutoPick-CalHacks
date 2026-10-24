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
without a new DRIVE, and emits status every 100 ms. Its lift completes instantly;
it is not a mechanical model or a substitute for B3/L2 bench testing.

Wheel speeds are mm/s. Lift targets are mm above the home switch, limited by
the configured presets. Status includes wheel step counts, lift mm, tray grams,
E-stop and a monotonic reception time. Nonzero driving requires fresh status,
E-stop released, and tray at zero with no lift command in progress. Homing is
required by the fake firmware before lifting. Every lift command first sends
STOP to the wheels. The real firmware must enforce
the same physical interlocks independently.

Replies and asynchronous STAT messages are separated by a background reader.
Requests serialize because there are no request IDs in the protocol. STOP
bypasses request waits. Serial faults, firmware ERR, malformed status and command
timeouts require reconnecting; this prevents late replies from being attributed
to a new command. Callers can still attempt STOP after a fault.

**Interface limitation:** the section 9.5 STOP command only specifies stopping
the wheels. It cannot satisfy the mission-wide lift-hold requirement by itself.
An approved firmware/interface decision is required before real M1/M2 STOP
integration. B4 does not invent a new serial command or claim lift-stop coverage.
Do not open real ports or move hardware without SW2 confirming the area is clear.
