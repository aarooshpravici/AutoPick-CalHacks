# B3 Uno R4 WiFi / 28BYJ-48 firmware (issue #47)

This replaces the earlier ESP32 STEP/DIR draft. The RA4M1 is the controller;
the R4's ESP32 WiFi coprocessor is not used. Four 5 V 28BYJ-48 motors each have
one ULN2003-style board. Left IN1..IN4 signals: D8, D9, D10, D11; right:
D2, D3, D4, D5. Two drivers on a side share inputs, not motor outputs. Motors
use the buck's 5 V rail, never GPIO supply. HW1 must check ratings and grounds.

**Do not energise the reported suspect driver boards.** Wiring is unfinished
and replacements may be needed. No upload or powered test has been performed.
`base.wiring_confirmed` stays false, so DRIVE is refused and coils stay off.
After repair and HW1/SW2 clearance, set it true, regenerate and build for upload.

The protocol stays at 115200 baud. Wheel speeds are mm/s and STAT counts emitted
half steps, not encoders. Stretch commands return `ERR no lift`; lift/weight
fields are zero. A 300 ms timeout since the last valid DRIVE stops stepping;
PING/malformed traffic cannot renew it. STOP decelerates quickly then replies
OK STOP. Both STOP and watchdog keep the last coil phase energised to hold;
physical power removal cannot hold position. Coils are off at boot.

Pin map, direction inversions, nominal 4096 half steps/rev, wheel geometry and
limits are generated from config/robot.yaml. Speed is provisionally capped at
30 mm/s (not the old 200), with 30 mm/s² normal and 300 mm/s² stop acceleration.
Measured loaded speed and turn rate are still required. Each loop emits at most
one half step per wheel, never blocks for a motor move, and bounds serial work.
Serial TX congestion may drop a reply; the Mac treats a missing reply as fault.

No E-stop sense wire is specified (`estop_sense_pin: -1`). STAT estop=0 therefore
means uninstrumented, NOT verified safe. The physical switch must cut motor
power, including the 5 V buck feed. If HW1 supplies an isolated/contact sense
wire, configure a free pin; never feed 12 V into the Uno pin.

Build without uploading:

```sh
.venv/bin/python base/firmware/generate_config.py
.venv/bin/python base/firmware/generate_config.py --check
.venv/platformio/bin/pio run -d base/firmware
```

The Apple Silicon build uses pinned GCC ARM 14.2.1 because the platform's default
GCC package is Intel-only. Arduino core 1.6.0 has a TinyUSB void-return diagnostic;
`-Wno-error=return-mismatch` retains it as a warning. The board build succeeds
(3328 bytes RAM, 45092 bytes flash); this is compile evidence, not bench acceptance.

PlatformIO Core 6.1.18; pinned `renesas-ra@1.9.0`, board `uno_r4_wifi`:
https://docs.platformio.org/en/latest/boards/renesas-ra/uno_r4_wifi.html

Native tests (no board):

```sh
c++ -std=c++17 -Wall -Wextra -Werror -I base/firmware/include base/firmware/test/controller_test.cpp -o /tmp/controller-test
/tmp/controller-test
c++ -std=c++17 -Wall -Wextra -Werror -I base/firmware/include base/firmware/test/stepper_test.cpp -o /tmp/stepper-test
/tmp/stepper-test
```

Bench checklist, ONLY after wiring repair and explicit area-clear approval:
1. Wheels raised: PING, STAT; low-speed DRIVE each pair; verify coil order/direction.
2. STOP and hold; unplug USB while driving and measure stopping <=300 ms.
3. On the loaded base, measure straight travel/time and turning without stalls.
4. Measure half steps/rev and effective track width; replace provisional limits.
5. Test physical E-stop cuts both arm and wheel motor rails. Record evidence in #47.

A compile/native test does not satisfy these checks. Slow/weak steppers may make
a staged, manually parked arm demo the practical fallback.
