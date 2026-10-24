from dataclasses import replace
import threading
import time
import unittest

from base.driver import BaseDriver, FakeSerial


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.port = FakeSerial()
        self.driver = BaseDriver(self.port, max_speed_mm_s=200, max_lift_mm=1000, timeout=0.25)
        self.addCleanup(self.driver.close)
        self.driver.status(timeout=1)

    def test_protocol_commands(self):
        self.driver.ping()
        self.driver.drive(50, -25)
        self.assertEqual((self.port.left, self.port.right), (50, -25))
        self.driver.stop()
        self.assertEqual((self.port.left, self.port.right), (0, 0))
        self.driver.lift_home()
        self.driver.lift_to(700)
        self.assertEqual(self.port.lift_mm, 700)
        self.port.weight = 125
        self.assertEqual(self.driver.tray_grams(), 125)
        self.driver.tare()
        self.assertEqual(self.driver.tray_grams(), 0)
        self.assertEqual(self.port.commands[:5], ["PING", "DRIVE 50 -25", "STOP", "LIFT_HOME", "LIFT_TO 700"])

    def test_limits_and_nonfinite_values(self):
        for speeds in ((201, 0), (0, -201), (float("nan"), 1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                self.driver.drive(*speeds)
        for height in (-1, 1001, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.driver.lift_to(height)
        self.assertEqual(self.port.commands, [])

    def test_drive_blocked_when_raised(self):
        self.driver.lift_home()
        self.driver.lift_to(700)
        with self.assertRaises(RuntimeError):
            self.driver.drive(10, 10)
        self.driver.stop()
        self.assertEqual(self.port.lift_mm, 700)  # STOP is wheels-only in frozen protocol.

    def test_stale_status_and_estop(self):
        with self.driver._condition:
            self.driver._status = replace(self.driver._status, received_at=time.monotonic() - 1)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                self.driver.drive(10, 10)
            self.driver._status = replace(self.driver._status, estop=True, received_at=time.monotonic())
            with self.assertRaises(RuntimeError):
                self.driver.drive(10, 10)
        self.driver.stop()  # Remains available even when motion is disallowed.

    def test_watchdog_without_renewal(self):
        now = [0.0]
        port = FakeSerial(clock=lambda: now[0])
        port.write(b"DRIVE 100 100\n")
        now[0] = 0.299
        port.readline()
        self.assertEqual(port.left, 100)
        now[0] = 0.3
        port.readline()
        self.assertEqual((port.left, port.right), (0, 0))
        self.assertAlmostEqual(port.left_steps, 30 * port.steps_per_mm)

    def test_fake_requires_lift_home(self):
        with self.assertRaises(RuntimeError):
            self.driver.lift_to(700)
        self.assertEqual(self.port.lift_mm, 0)

    def test_partial_lines_and_unsolicited_status(self):
        with self.port._lock:
            self.port._lines.extend([b"STAT 12 ", b"34 0 9.5 0\n"])
        deadline = time.monotonic() + 1
        while self.driver.status().left_steps != 12 and time.monotonic() < deadline:
            time.sleep(0.001)
        status = self.driver.status()
        self.assertEqual((status.left_steps, status.right_steps, status.tray_g), (12, 34, 9.5))
        self.driver.ping()

    def test_stop_bypasses_waiting_lift_request(self):
        original_write = self.port.write
        requested = threading.Event()

        def delayed_write(data):
            if data == b"LIFT_HOME\n":
                requested.set()
                return len(data)
            return original_write(data)

        self.port.write = delayed_write
        failures = []

        def home():
            try:
                self.driver.lift_home()
            except TimeoutError as error:
                failures.append(error)

        thread = threading.Thread(target=home)
        thread.start()
        self.assertTrue(requested.wait(1))
        start = time.monotonic()
        self.driver.stop()
        self.assertLess(time.monotonic() - start, 0.1)
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        with self.assertRaises(RuntimeError):
            self.driver.ping()  # Do not mistake a late response for a later command.

    def test_close_stops_wheels_and_closes_thread(self):
        self.driver.drive(10, 10)
        self.driver.close()
        self.assertTrue(self.port.closed)
        self.assertEqual(self.port.commands[-1], "STOP")
        self.assertFalse(self.driver._reader.is_alive())
        with self.assertRaises(RuntimeError):
            self.driver.drive(10, 10)


if __name__ == "__main__":
    unittest.main()
