# M1 mission simulation

The brain implements the `ui.server.Brain` contract: thread-safe immediate
`request`, nonblocking `stop`, cached `status`, and no thumbnails. A 10 Hz loop
runs Search → Approach → Pick → Present → WaitForTake → Idle. It calls the real
`pick_to_hand` function with `SimArm`, its cancellation event and state callbacks.
`FakeSerial` supplies base status and the wheel watchdog. No lift is assumed;
UI lift/weight fields are null.

From this checkout, use the Python 3.12 robot environment:

```sh
.venv/bin/python -m brain.mission
.venv/bin/python -m ui.server --sim
```

The first command runs one complete mission and exits with 0 only for `taken`.
The second serves the existing status page; request a phone there. Simulation
uses virtual arm time, so arm states may pass faster than page polling; every
transition is recorded in `data/logs/<date>.jsonl`. JSONL also includes requests,
detections, serial commands and arm results. No camera is opened.

To consume the P3 publisher from PR #32, use two terminals (no sudo):

```sh
# Terminal A, checkout containing P3:
.venv/perception/bin/python -m perception.publish --fake
# Terminal B, checkout containing M1:
.venv/bin/python -m brain.mission --udp
```

Only one listener can own `perception.detections_udp`; stop `perception.listen`
before this test. `--replay data/logs/p3-live.jsonl` reads saved 9.2 messages,
one per tick with refreshed replay timestamps. Malformed/stale UDP traffic does
not refresh camera health; a camera gap over one second stops the mission.
Five distinct matching frames with confidence >=0.6 are required to approach.

Navigation is explicitly **simulated**: 24 synthetic turn/pause steps represent
two turns, and five approach steps represent parking. It accepts null floor
coordinates from P3 fake mode. These stand-ins do not establish real geometry,
search performance or parking accuracy. `build(config, sim=False)` refuses
before opening hardware until measured P4/B5/B6 controllers are integrated.

STOP sets the arm cancellation event and the latched Stopped state immediately.
One serial worker handles both drive and stop, prioritising STOP and discarding
old mission commands. UI methods never hold a lock during I/O. A replaced
mission waits for the old arm worker to finish holding before driving/picking.
The fake base receives STOP before the arm pick begins. B4 serial failures are
reported; a software test does not establish the real 0.5-second STOP timing.

A dropped object retries the full search/pick at most three times. Missed,
stuck and timeout results enter Failed, then Idle after reporting for one second.
`stopped` stays latched until a new request. A timeout leaves the arm holding at
carry, as specified by the arm API. Base/logging failures inhibit new requests.

M1 is stacked on B4 PR #33 for the wheels-only fake and approved STOP contract.
Tests cover the actual SimArm success/timeout/miss/drop/stuck paths, STOP during
hand-off, STOP while serial I/O blocks, replacement races, camera loss, empty
search exhaustion, UDP/replay and existing UI routes. No real motion is enabled.

## Physical prerequisites for the next tasks

- **P4:** secure the D435 on its final low mount (about 7–12 cm high, 20 degrees
  down), keep the image size/flip fixed, and finish SW1's on-base grab poses.
  Lay printed 10 cm ArUco markers flat at measured positions. Supply the phone
  at the actual reachable grab point; keep three separate measured test points
  to verify <=2 cm mapping error. A moved camera invalidates calibration.
- **B5:** P4 must pass, HW1 must confirm motor/wiring/dimensions and E-stop, and
  B3 bench drive/watchdog checks must pass. Clear a floor area, mark the grab
  spot and have someone at E-stop. SW2 must explicitly clear motion before
  <=100 mm/s first runs and ten measured parking trials.
- **B6:** first pass B5; use the same fixed mount, a clear turning footprint,
  and phones within 1 m at varied headings. Confirm skid turning does not stall,
  keep someone at E-stop, and explicitly clear motion before search trials.

These are prerequisites, not claims that the hardware gates have passed.

### Measured navigation (issue #47 draft)

Real UI mode uses `build(config, sim=False)` only after P4's measured calibration,
recorded parking target and on-base handoff poses exist. HW1 must finish wiring,
verify wheel directions, measure loaded speed and turning, then record
`base.wiring_confirmed` and `base.drive_calibrated`. Defaults remain false.
A person must clear the area before starting real mode. The agent has not run it.
The existing UI real-mode confirmation still applies.

Search uses commanded step counts, which cannot detect missed steps or skid slip.
Parking stops on stale or invalid floor data and requires three fresh settled frames.
The firmware must acknowledge STOP before the arm starts. Missing acknowledgment
fails the mission. First real scripted picks run at no more than 25 degrees/second.
The wrist-roll exclusion remains in force; no wrist camera or ACT model is required.

Physical acceptance still requires B5 8/10 parking trials, B6 full-heading search,
five complete M2 runs and measured STOP response within 0.5 s, including midmission.
Simulation results do not satisfy these checks.
