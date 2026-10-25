# SW2 software handoff for the phone demo

Integration branch: **`task/HP-software`**. This combines the existing open PRs
through #59, including #57's distance form/firmware and #54's suction support.
SW2 tested software only. No board was flashed, no real port/camera was opened,
and no hardware motion or calibration was performed. Keep Phone as the sole
selectable object; remote/sock detection is not ready for this demo.

Use the reviewed integration checkout on SW1's robot Mac. Preserve its local
`config/poses.json`, `config/floor_homography.json`, model weights, and measured
configuration; do not overwrite them with another Mac's placeholders. Check
`git status` before switching branches, and save local configuration changes.
The generated firmware header must match the actual configuration before flash.
Do not replace the existing LeRobot installation just to run these checks.

## Read-only software checks (no sudo, no devices opened)

```sh
cd "$HOME/Desktop/Cal Hacks/AutoPick-CalHacks"
/opt/miniconda3/envs/lerobot/bin/python -m brain.check --role robot
/opt/miniconda3/envs/realsense/bin/python -m brain.check --role perception
```

Each prints JSON and exits 1 for missing prerequisites. Missing poses, weights,
floor evidence, wiring flags or dependencies are reported individually. Exit 0
checks stored software/files only; it does not verify wiring, traction, camera
mount, physical E-stop or pickup success. The perception role requires Python
3.10, RealSense SDK, Ultralytics, NumPy and OpenCV. The robot role requires Python
3.12 and LeRobot 0.6.2. MediaPipe is not needed for the phone-only baseline.
Use `perception/requirements.txt` to resolve missing packages in the perception
environment after reviewing that environment's existing pins.

The non-moving link checker is a separate **person-run USB check**:

```sh
/opt/miniconda3/envs/lerobot/bin/python -m base.link --list
/opt/miniconda3/envs/lerobot/bin/python -m base.link --port /dev/cu.REPLACE_WITH_R4_PORT
```

It sends only PING/STOP. Opening USB may reset the controller; use motor power
off for the first link check. Keep other serial programs closed. STAT estop=0
is not proof of a working physical E-stop when `estop_sense_pin: -1`.

## Software-only rehearsal

```sh
/opt/miniconda3/envs/lerobot/bin/python -m pytest -q
/opt/miniconda3/envs/lerobot/bin/python -m brain.mission --item phone
/opt/miniconda3/envs/lerobot/bin/python -m base.jog --direction forward
/opt/miniconda3/envs/lerobot/bin/python -m ui.server --sim --host 0.0.0.0 --port 8000
```

Check pytest's actual exit status. No pipe to tail. The first three commands exit;
the last serves the simulated website. Enter feet, select Phone/height and Start
request. Test STOP and Escape in Search, Approach and handoff. Other devices use
SW1's printed LAN address, not their own localhost. This has no login, so use a
trusted network. The fake jog proves protocol ordering, not physical direction.

## Grasp selection is now connected to the brain

Default `handoff.grasp: claw` preserves existing behavior. For the phone's
passive cup, a person first installs/tests the cup and teaches the suction poses;
then set local `handoff.grasp: suction`. The defaults are
`suction_dwell_s: 2` and `suction_hold_s: 4`; allowed ranges are >0–10 s and
>0–60 s. These settings do not alter arm speed or step caps.

The brain loads suction-mode poses and passes the selected mode to `pick_to_hand`.
A finished passive-cup sequence reports **unverified**, not “Item taken”: there
is no seal/take sensor. The person must visually confirm the object was lifted
and taken. STOP still interrupts the sequence and holds the arm. The real path
requires wrist_roll frozen, strict verified base flags, an R4 port different
from either arm port, validated floor/parking evidence and taught poses.

Person-run teaching and independent pick checks, after line-up/area checks:

```sh
/opt/miniconda3/envs/lerobot/bin/python -m arm.teach --speed 25 --grasp suction
/opt/miniconda3/envs/lerobot/bin/python -m arm.pick run --speed 25 --grasp suction
```

Do not use bench poses as on-base acceptance. Never use a direct LeRobot follower
CLI or old safe_teleop that includes faulty servo 5. Do not recalibrate it.

## Bounded wheel check for Aroosh (a person runs this, never an agent)

After HW1 verifies actual wiring/power and the configuration/header is reviewed,
regenerated and flashed on the R4, close the robot server and serial monitor.
Raise the wheels for first tests. Keep the area clear and a hand at the E-stop.

```sh
/opt/miniconda3/envs/lerobot/bin/python -m base.jog --real --port /dev/cu.REPLACE_WITH_R4_PORT --direction forward
```

This refuses noninteractive execution and asks for Enter **before opening USB**.
It acknowledges STOP, sets a small firmware LIMIT, drives at 5 mm/s for at most
1 second / 10 mm mean wheel travel, then waits for STOP. Ctrl-C/errors also
attempt STOP. It runs one direction only. Repeat manually with `back`, `left`,
`right`, checking observed direction each time. Caps are 10 mm/s (or a lower
configured cap), 2 seconds and 20 mm; drive calibration is deliberately not
required for this initial test, but verified wiring is. No automatic repeat.

A firmware LIMIT is required after boot; STOP/watchdog preserve its remaining
budget. Firmware and driver must be deployed together. An older firmware refuses
LIMIT before DRIVE. A failed STOP is not success: use the physical E-stop.
Do not enable `drive_calibrated` for M1 until actual loaded checks pass.

## Perception / floor calibration (person-run camera access)

Stop other camera programs. Ensure the local `data/models/phone_v2.pt` is the
reviewed phone checkpoint (not committed here). A person types sudo:

```sh
sudo /opt/miniconda3/envs/realsense/bin/python -m perception.publish --device mps --preview
```

Before running the brain, use another terminal for placement measurements:

```sh
/opt/miniconda3/envs/realsense/bin/python -m perception.listen --placement 0.5m-centre --label phone
```

The listener and brain both bind port 5601, so stop the listener before starting
the brain. Measure the other distances/left-centre-right and empty floor too;
reports go to ignored `data/logs/p3-live.jsonl`. A preview alone is not measured
floor calibration or a reliable pickup target.

P4 needs **four fit markers plus three independent check markers**, all measured
on the final fixed floor/camera setup. Generate the marker/layout files with:

```sh
/opt/miniconda3/envs/realsense/bin/python -m perception.calibrate_floor markers
```

Follow `perception/README.md`'s P4 steps for measurements, collection, fitting,
then SW1's reachable phone bottom-centre parking pixel. Use the realsense
interpreter above in place of `.venv/perception/bin/python`; a person types sudo
for collection. All independent errors must be <=2 cm. The 0.33 m reach guard
uses the floor origin below the camera: verify the camera-to-arm offset on the
actual base rather than treating that estimate as a measured reach envelope.

## Real end-to-end handoff remains a human test

Only after the local setup report and physical prerequisites are satisfied,
leave the live publisher running and have a person start:

```sh
/opt/miniconda3/envs/lerobot/bin/python -m ui.server --real --host 0.0.0.0 --port 8000
```

It asks for area-clear confirmation. Use the printed URL. Enter feet, select
Phone/height, then Start request. Search turns consume the same estimated wheel
budget as approach. At the budget limit, the base holds, the brain checks a
new camera frame and stays stopped; it never auto-returns or silently adds
travel. At the measured pickup spot it needs a fresh STOP acknowledgment before
starting the arm, including after any resumed approach.

Send back: exact commit, both preflight JSON outputs, link-check JSON, observed
wheel directions/STOP results, live detection log and whether the phone pickup
was visually successful. Do not claim the autonomous demo passed from unit
tests or an unverified suction result. A labelled teleop video remains a fallback.
