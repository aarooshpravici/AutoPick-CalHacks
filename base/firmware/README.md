# B3 wheels-only ESP32 firmware

Implements plan 9.5 at 115200 baud. Left/right commands each control a pair of
wheels. `STAT` is emitted every 100 ms; lift/weight fields are zero. All stretch
commands return `ERR no lift`. A valid DRIVE must arrive at least every 300 ms;
the watchdog cuts pulse generation at that deadline. PING and invalid commands
do not renew it. STOP ramps down quickly, and E-stop stops immediately. Release
of E-stop does not resume an old command. Malformed commands stop motion.

**Not configured for the physical robot yet.** `include/pinmap.h` has disabled
outputs and -1 pins. The provisional backend is STEP/DIR for paired stepper
drivers; use it only if HW1 confirms that motor/driver type. DC motors require a
matching H-bridge backend (and calibrated speed control/encoders), not these
signals. Confirm the board model too: esp32dev is a compile target, not a claim
about Darren's board. Until confirmed, DRIVE returns `ERR unconfigured motors`.

HW1 must provide the motor/driver models, pin map, polarity, pair wiring,
microsteps, wheel diameter, track and wheelbase, and isolated E-stop sense
contact. Never feed 12 V into a GPIO. Check usable pins, boot-strapping and
input-only pin restrictions against the actual board before setting confirmed.
No pin values have been guessed. Both left drivers share one step/direction
signal; both right drivers share another. STAT counts emitted pulses, not
encoder feedback; it cannot detect wheel slip or a stalled motor.

Build only (no upload):

```sh
python3.12 -m venv .venv/platformio
.venv/platformio/bin/pip install platformio==6.1.18
.venv/platformio/bin/pio run -d base/firmware
```

The platform is pinned to espressif32 6.12.0 with Arduino. PlatformIO reference:
https://docs.platformio.org/en/latest/platforms/espressif32.html

Native protocol checks, no board or PlatformIO required:

```sh
c++ -std=c++17 -Wall -Wextra -Werror -I base/firmware/include \
  base/firmware/test/controller_test.cpp -o /tmp/autopick-firmware-tests
/tmp/autopick-firmware-tests
```

Serial RX work is bounded per loop, lines are bounded to 95 bytes, and TX is
only attempted when buffer space is available, so a serial flood cannot block
the watchdog. A congested TX buffer may drop replies/status; the Mac must treat
missing responses as a fault. The loop generates at most one pulse per wheel
per iteration; verify achievable speed and watchdog timing on the actual board.

Bench acceptance needs SW2's explicit area-clear go-ahead and HW1's checked
power/E-stop wiring. First lift the wheels clear, verify direction for each
pair at <=100 mm/s, STOP, and USB-loss stopping within 300 ms. Then check turning
in place on the actual floor for scrubbing/stalling. Record measured timings;
native tests and a successful compile do not satisfy these hardware checks.
Lift firmware/load-cell calibration remain stretch work, not implemented here.
