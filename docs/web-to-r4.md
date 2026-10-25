# Website → Arduino Uno R4

Run the robot server on **SW1/Aroosh's Mac**, which owns the R4 USB connection,
arm connection and camera. Open that Mac's website from the other laptop/phone.
A page served by SW2's localhost cannot reach a USB cable on SW1's Mac.

```text
Browser on either laptop/phone
    Form → POST /fetch {"item":"phone","hand":"standing","distance_ft":3.5}
    text  → POST /request {"text":"I dropped my phone","hand":"seated"}
                     ↓
SW1 Mac: FastAPI → Mission → camera-guided search/parking → BaseDriver
                     ↓ USB serial, 115200 baud
Uno R4: LIMIT <mm> / DRIVE <left_mm_s> <right_mm_s> / STOP
                     ↓
Four wheel motors
```

The distance-and-object form bypasses the language model. Text requests retain the existing parser.
The R4 receives bounded wheel commands, not natural language or arbitrary code.
The Mac refreshes commands while navigating. Firmware's 300 ms watchdog stops on
lost DRIVE commands; STOP decelerates and replies `OK STOP` before pickup starts.
The SO-101 arm is controlled separately by the Mac; it is not controlled by this
R4 wheel firmware. Lift/load-cell support is not fitted in the current baseline.

## First check the USB link on SW1's Mac (no wheel movement)

Use the reviewed integration branch/merged code including B3 #34, P3 #32,
P4 #48, B5 #50, B6 #51, M1 #52 and UI #56. Keep SW1's actual ports, calibration
IDs, poses and frozen wrist-roll setting. Do not replace measured local files
with placeholder values from another checkout. Close Arduino Serial Monitor or
any other process holding the R4 port.

From the checkout root, using the Python 3.12 environment:

```sh
.venv/bin/python -m base.link --list
```

Select the R4's listed `/dev/cu...` port explicitly, then:

```sh
.venv/bin/python -m base.link --port /dev/cu.REPLACE_WITH_R4_PORT
```

This sends **CAN/STOP synchronization, PING and INFO**, waits for PONG/STAT/OK STOP and reports a
round-trip time. It does not send DRIVE, flash firmware, validate motor wiring,
or measure physical stopping time. The board must already run the reviewed R4
firmware. Opening USB can reset a microcontroller; perform the initial link check
with the motor supply off. Do not guess an arm's serial port is the R4.

## Network preview before enabling hardware

On SW1's Mac, start the same website with fake hardware:

```sh
.venv/bin/python -m ui.server --sim --host 0.0.0.0 --port 8000
```

The server prints its URLs. On SW2's Mac/phone, open
`http://<SW1-MAC-LAN-IP>:8000` on the same trusted network. Use the actual LAN
address, not `localhost` or `0.0.0.0`. Venue networks sometimes isolate clients;
if the page will not load, use a shared trusted network or hotspot and allow
Python's incoming connection through macOS's firewall. This page has no login;
only expose it to trusted devices, not the public internet.

## Real fetch, after the physical checks

Finished wiring is reported by SW2. Still verify the installed firmware, wheel
polarity, loaded speed/turning, physical E-stop and measured STOP response.
Record the actual R4 `base.serial_port`, verified drive limits and completed
`wiring_confirmed`/`drive_calibrated` checks. Regenerate the firmware pin map after
configuration changes; B3's default firmware refuses DRIVE until configured.

P4 must provide `config/floor_homography.json`, the recorded reachable parking
target and matching camera size/flip. SW1's on-base `config/poses.json` must be
valid for handoff. Keep `arm.frozen_joints: [wrist_roll]`. Real mode refuses to
open hardware if its prerequisite checks fail. The approximate 33 cm reach
estimate is not a replacement for a measured final-base pickup position.

With the connected camera available, run its publisher separately on SW1's Mac
(Python 3.10 perception environment):

```sh
sudo .venv/perception/bin/python -m perception.publish --device mps
```

Once a person has cleared the area and is ready at the E-stop, run the robot
server (Python 3.12, LeRobot 0.6.2), answer its real-hardware confirmation, and
open its printed LAN URL:

```sh
.venv/bin/python -m ui.server --real --host 0.0.0.0 --port 8000
```

Tap Phone. Mission status should show search/approach; the R4 receives DRIVE
updates until parked, then STOP. Pickup waits for OK STOP. Page STOP/typed stop
request cancellation; serial faults are reported, not silently replaced with
simulation. A live physical run and timing measurement are still required.

## Local proof without any hardware

`tests/test_web_r4.py` connects the web app, actual mission/navigation and
pyserial to a pseudo-terminal running the firmware's C++ `Controller` core.
It checks direct clicks and text requests, bounded DRIVE commands, STOP-before-
pick acknowledgment and STOP during a mission. Camera/geometry/arm are simulated;
commanded steps are not real odometry. These tests do not claim a physical run.

```sh
.venv/bin/python -m pytest -q tests/test_web_r4.py
```

## Distance form and R4 travel budget

The page now asks **“How far in feet did you drop your item?”**, then requires
Phone selection and retains Standing/Seated. **Start request** posts
`{"item":"phone","hand":"standing","distance_ft":3.5}` to `/fetch`.
The six-step tutorial explains distance, object, height, start and STOP.
Accepted estimates are finite numbers greater than zero, up to 20 feet.

Mission acknowledges STOP, sends `LIMIT 1066.800000` for 3.5 ft, waits for
`OK LIMIT`, then starts existing camera-guided search/approach. This does **not**
mean blindly driving 3.5 ft forward: there is no direction input. Search turns
also consume the budget. A reachable, measured parking spot permits an earlier
pickup. The 33 cm pickup guard is independent of the user's travel estimate.
Its provisional origin is the floor below the camera; SW1 must verify the
camera-to-arm offset and the measured target on the final base.

The R4 counts mean absolute left/right issued half-step travel, including turns
and reversals. It stops one half-step before exhausting the budget, keeps coils
energized and emits `LIMIT REACHED`. Further DRIVE cannot restart it; STOP and
the watchdog preserve the budget. A new acknowledged LIMIT while stopped is
needed for another mission. Boot requires LIMIT before any DRIVE.
These step counts do not measure wheel slip or real floor displacement.

On exhaustion the brain waits for STOP acknowledgment, then a newly captured
RealSense frame. It reports item seen, item not found, or no fresh frame after
2 seconds, and remains **Stopped**. It does not start an arm pick, drive past
the estimate, reverse or turn again automatically. Reposition and submit a new
request after checking the area. Existing camera loss, STOP and search/approach
timeouts can stop a mission earlier.

Deploy this firmware, driver and brain together on SW1's Mac/R4. Older firmware
rejects LIMIT: the mission fails closed before DRIVE. Legacy `/request` and
`/fetch` callers omitting distance get a 20 ft default; the visible form always
requires an explicit estimate. No real motion or flash was performed by SW2.
