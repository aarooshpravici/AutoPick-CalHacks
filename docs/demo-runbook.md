# Demo-video runbook: Sunday 25 October

**Goal:** a 5-minute video of AutoPick, **exported by 08:15**. Shoot from about 07:30, then edit.

This runbook comes after HIGH-PRIORITY.md H1–H7. If H7 (one clean end-to-end run) isn't done by 07:00, start this runbook anyway and use the cover-ups in section 4.

| Time | What |
|---|---|
| 07:00 | Pre-flight (section 1), about 20 minutes |
| 07:20 | One rehearsal run, not filmed |
| 07:30–07:55 | Shoot (section 3), with retakes |
| 07:55–08:15 | Cut, add captions, export |

## Rules that don't bend tonight

- **A person** runs every command that moves the arm or the wheels, and types the sudo password. Agents never do.
- **Darren keeps a hand on the E-stop in every take.** Before anything moves, the area is clear.
- **STOP must work for real:** the page button, Esc on the page, and the E-stop. Never weaken a speed cap, the 300 ms watchdog, the travel LIMIT, the 33 cm reach guard or STOP to make a take work.
- **The follower's wrist roll (servo 5) stays frozen** (`arm.frozen_joints: [wrist_roll]`).
  - Only use `arm.teach`, `arm.pick`, `arm.teleop` and `ui.server`. Never use `lerobot-teleoperate`, `lerobot-record`, `lerobot-calibrate` or `safe_teleop.py`.
  - Never pass `--roll`.
- **One program per port.** Each arm port, the R4 port, the RealSense and UDP port 5601 can each be used by only one program at a time. Stop the server before any arm-only command, `bus_health.py` or `base.link`.

## Who does what during the shoot

| Who | Job |
|---|---|
| **SW1** | Arm safety. Stands beside the robot, out of frame, with a hand near the arm's 12 V switch. Runs the arm-only commands for the cover-ups and resets the arm between takes. |
| **SW2 (Juan)** | Starts perception and the server, then acts as the person in the video: makes the request on the phone and takes the object. Calls "cut" if a terminal shows an error. |
| **HW1 (Darren)** | Hand on the E-stop in every take, out of frame. Puts the robot back on its tape mark. |
| **HW2** | Camera. A phone on a tripod, landscape, 30 fps, side-on at the robot's height. Also screen-records the Mac (QuickTime: File → New Screen Recording). |

---

## 1. Pre-flight checklist (about 20 minutes)

**Code and configuration**

- [ ] SW1's checkout is on the branch SW2 chose in H2: `main` after the merge, `task/HP-software` (#60) or `integration/demo`.
  - Check it out **before** editing `robot.yaml`.
  - Don't `git pull` again until the video is done.
  - The local files stay: `config/poses.json`, `config/floor_homography.json` and `data/models/phone_v2.pt`.
- [ ] `config/robot.yaml` holds the measured values. `ui.server --real` refuses to start without them:
  - `base.serial_port`: the R4's port, not an arm port;
  - `base.invert_left` / `invert_right`, as found in H3;
  - `base.wiring_confirmed: true`, and `base.drive_calibrated: true` (only once H3's loaded checks have passed);
  - `perception.parking_spot_floor_m`, written by `calibrate_floor parking` in H6;
  - `handoff.grasp: claw` or `suction` (needs #60);
  - `arm.frozen_joints: [wrist_roll]`, unchanged.
- [ ] The R4 runs firmware built from **this** `robot.yaml`. If any `base:` value changed after the flash, the firmware has to be regenerated and re-flashed (H3; see `base/firmware/README.md`).

**Power and E-stop**

- [ ] Power is on:
  - the arm's 12 V supply;
  - the steppers' 5 V buck;
  - the R4 on USB to the Mac.
- [ ] The Mac is on a charger, or above 60%. Tethered or wall power is fine.
- [ ] The leader's 5 V is plugged in, needed only for a teleop take. If it shows 0 motors, reseat the plug.
- [ ] **E-stop test**, with the arm folded at `ready` and SW1 holding it:
  - press the E-stop: the arm's torque and the wheel power both go off;
  - release it;
  - then **restart the server** (section 2, step 9).
  - `STAT estop=0` proves nothing: there is no E-stop sense wire.

**Checks (section 2, steps 1–6):**

- [ ] `arm.check`: exit 0.
- [ ] `bus_health.py`: 0/100 failures on the follower (and on the leader, for teleop).
- [ ] `brain.check --role robot`: every check ok (needs #60).
- [ ] `base.link`: `pong` and `stop_acknowledged` are both true. Run it with the wheels off the ground or the motor power off.

**Camera and page**

- [ ] **Camera:**
  - The RealSense is on its own USB 3 port, and no other camera app is open.
  - The preview window shows a green box on the phone at the start spot, and the publisher prints `floor mapping=True`.
  - `perception.listen` shows `hz` ≥ 10.
- [ ] **Page on the phone:**
  - The phone is on the same Wi-Fi as the Mac, or both are on one phone's hotspot. Auto-lock is off.
  - The header says **Robot control**, not Simulation.
  - Tap STOP with nothing moving: the state goes to Stopped.
- [ ] **The area:**
  - 2 m × 2 m of clear floor: the same floor and camera mount as the calibration.
  - Tape an X for the robot's start, and a mark for the object about 0.6–0.8 m straight ahead.
  - Bright, even light. The cables are slack and can't catch.
- [ ] **Recording:** the tripod phone has free space, and the Mac's screen recording is ready.

## 2. Start-up commands, in order

Run them on SW1's Mac. Each block is one command; paste it as it is.

**Terminal 1: checks (read-only, nothing moves)**

1. Which branch and commit, and only the expected local files:

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && git status --short && git log --oneline -1
```

2. Arm pre-flight. It reads files only. For the cup, add `--to suction`.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m arm.check
```

3. Servo bus health (read-only). Nothing else may have the arm ports open.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks" && /opt/miniconda3/envs/lerobot/bin/python bus_health.py
```

4. Robot pre-flight. It needs #60; skip it if `brain/check.py` doesn't exist. It opens no device.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m brain.check --role robot
```

5. List the serial ports. The R4 is the `usbmodem` that is **not** `58CD1767761` (follower) or `58760435221` (leader).

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m base.link --list
```

6. R4 link check: PING and STOP only, never DRIVE. Use the same port as `base.serial_port`.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m base.link --port /dev/cu.usbmodemREPLACE_WITH_R4
```

**Terminal 2: perception (sudo: the person types the password)**

7. The camera and phone detector, with the preview window. Leave it running. Q in the window, or Ctrl-C, stops it.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && sudo /opt/miniconda3/envs/realsense/bin/python -m perception.publish --device mps --preview
```

**Terminal 3: detection check, then the server**

8. A 5-second detection check with the phone on its mark. It exits by itself. Look for `detected_frames` > 0 and `hz` ≥ 10. It must finish before step 9, because both use port 5601.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m perception.listen --placement demo-start --seconds 5
```

9. The robot server. It asks "Is the area clear and is someone at the E-stop?". Type `yes` only when that's true. Leave it running for every take.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m ui.server --real --host 0.0.0.0 --port 8000
```

10. The Mac's Wi-Fi address. On the phone, open `http://<that address>:8000`. The server also prints a `.local` link that may work.

```bash
ipconfig getifaddr en0
```

**Backup STOP from the Mac keyboard (any terminal)**

```bash
curl -s -X POST http://127.0.0.1:8000/stop
```

**Between takes**

- **After a take:**
  - the arm goes back to `ready` by itself;
  - put the robot on its X and the object on its mark;
  - start a new request on the page.
- **After STOP:** the state stays Stopped until the next request. The arm holds where it stopped, so do the STOP shot while the robot drives, with the arm at `ready`.
- **After the E-stop, "brain fault; reconnect before requesting", or a serial error:**
  - SW1 holds the arm;
  - Ctrl-C the server;
  - run step 6 again;
  - restart with step 9.
- **Shutting down:**
  - the arm at `ready`;
  - Ctrl-C the server, then the publisher;
  - SW1 holds or rests the arm before its 12 V goes off.

**Arm-only commands for the cover-ups.** Stop the server first, because these open the follower port.

Scripted claw pick → present → take, from the parking spot:

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m arm.pick run --speed 25
```

The Octobuddy cup on the phone. It needs the suction poses (`arm.teach --grasp suction --speed 25`). The hold is timed (4 s by default): the person takes the phone during it.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m arm.pick run --speed 25 --grasp suction
```

Teleop, for a labelled teleop take only:

1. The live line-up table shows both arms.
2. Press Enter only once every joint shows ✓ and the leader and follower match.
3. Ctrl-C stops it, and the follower holds.

```bash
cd "/Users/aarooshkurchania/Desktop/Cal Hacks/AutoPick-CalHacks" && /opt/miniconda3/envs/lerobot/bin/python -m arm.teleop --speed 25
```

## 3. The story and the shot list (5:00 at most)

**The story:** someone drops their phone and can't easily reach the floor. They ask on their phone. The robot finds it, drives up, parks, picks it up and holds it up to them. They take it. STOP works at any moment.

**On the page:** enter the distance in feet, choose Phone and the hand-off height, then tap **Start request**.

- For the distance, enter about 1 ft more than the real distance (3–4 ft for a phone 0.6–0.8 m away), because search turns also use the travel budget.
- When the budget runs out, the robot stops and stays Stopped. Retake with a bigger estimate.

| # | Time | Shot | What happens | The page shows |
|---|---|---|---|---|
| 1 | 0:00–0:25 | Hook | SW2, seated, drops the phone on the mark in front of the robot. Voice-over: the problem. | — |
| 2 | 0:25–0:45 | Request | A close-up of the phone page: feet, Phone, height, **Start request**. iPhone screen recording is the cleanest. | Search |
| 3 | 0:45–1:45 | Finds and approaches | Wide side shot. Cut-in: the Mac's preview window with the green box on the phone. The base creeps (about 20 mm/s). Speeding up is fine **if the caption says "2×"**. | Search → Drive |
| 4 | 1:45–2:00 | Parks | Low close-up: it stops with the phone at the arm's pickup spot. | Pick |
| 5 | 2:00–3:00 | Arm picks and presents | The arm reaches down, grips, lifts and presents at 25°/s or less. | Pick → Present → Take |
| 6 | 3:00–3:20 | The person takes it | SW2 pulls it out once the arm has stopped. The claw feels the pull, opens and goes back to `ready`. | Item taken |
| 7 | 3:20–3:50 | STOP works | Start a new request. While it drives, tap **STOP**: the wheels stop at once and the page shows Stopped. Then, with the arm folded at `ready` and SW1 holding it, Darren presses the E-stop. | Stopped |
| 8 | 3:50–4:30 | How it works | B-roll and a diagram (below). | — |
| 9 | 4:30–5:00 | What's next, and the team | Future work (section 5), names, thanks. | — |

**Shot 8, how it works:**

- RealSense and a YOLO phone detector, mapped to the floor;
- the brain on the MacBook;
- the Uno R4 and the stepper wheels;
- the SO-101 arm;
- safety: STOP, a 300 ms wheel watchdog, speed caps and a 33 cm reach guard.

**If you use suction (#60, `handoff.grasp: suction`):**

- The page ends with "visually confirm pickup and take", not "Item taken".
- Say "a timed hand-off", not "it felt the take".

## 4. Cover-ups: what's fine, and what we never fake

**Always fine:**

- cut and retake;
- start close (0.6–0.8 m) and facing the object;
- tethered or wall power;
- footage sped up with a "2×" caption;
- two segments with honest captions.

| If this goes wrong | Honest fallback for the video | Don't |
|---|---|---|
| **Detection is flaky** (the box flickers, or it never leaves Search) | 1. Start facing the phone at 0.6–0.8 m, on the calibrated floor, in brighter light, with the phone face up.<br>2. Stop the server, run step 8 until it detects, then start the server again (step 9).<br>3. Retake. | Don't change `detector_conf`, the 5-frame rule or the camera mount at 07:30 (moving the camera breaks the floor calibration). Don't push the robot by hand while calling it camera-guided. |
| **Parking misses** (stops short or long, or "target passed; reposition manually") | 1. STOP, put the robot back on its X, retake.<br>2. If it keeps missing: place the robot at the pickup spot by hand, stop the server, and run the arm-only pick.<br>3. Caption it "robot placed at the pickup spot". | Don't nudge the object into the claw in a take shown as autonomous. Don't raise `pickup_max_distance_m` (0.33). |
| **The claw misses the phone** (the fingertips push it away) | **Sock or remote:** the detector knows only the phone (it was trained with remotes as negatives) and the page offers only Phone. So film two captioned segments: (1) the mission finds and approaches the phone; (2) "arm pick and hand-off, shown with a sock", using the arm-only claw command from the parking spot.<br>**Octobuddy cup:** with #60, set `handoff.grasp: suction` and run the whole mission. Without #60, use the arm-only suction command. | Don't say the claw picked up the phone. Don't say the cup senses the seal or the take. Don't claim the sock was detected. |
| **The base drifts, turns badly or stalls** | 1. Start square to the object so no search turn is needed.<br>2. Start closer; use the tape X.<br>3. Retake. | Don't raise `max_speed_mm_s` or the acceleration, and don't touch the watchdog or the LIMIT. |
| **A servo browns out** (the arm goes limp or jerks, a `stuck` result, bus errors) | 1. SW1 holds the arm; E-stop, or switch off the arm's 12 V.<br>2. Reseat the plugs and power the arm back on.<br>3. Run `bus_health.py` (step 3), restart the server (step 9) and retake. | Leave the wrist roll alone: it stays frozen. Don't "fix" the arm with lerobot-calibrate, lerobot-teleoperate or `safe_teleop.py`. |
| **The page won't load on the phone** | 1. Put the Mac and the phone on one phone's hotspot.<br>2. Allow Python's incoming connections if macOS asks.<br>3. Use the address from step 10, not `localhost`. | Don't serve the page on the public internet: it has no password. |
| **Nothing autonomous works** | 1. Place the robot at the pickup spot by hand.<br>2. Stop the server.<br>3. SW1 runs `arm.teleop --speed 25`.<br>4. Caption the whole segment **"Teleoperated"**. | Never present a teleop take as autonomous, or cut it so that it looks autonomous. |

**Never fake:**

- STOP: it is a real tap, and the robot really stops;
- autonomy: no hidden hands moving the robot or the object;
- the learned grasp: ACT isn't running tonight.

## 5. What we say is future work

- **The ACT learned grasp:**
  - the native recorder is ready (#46);
  - next, record 50 demos and train on the lab PC;
  - it replaces the scripted grasp only if it wins 7 of 10 picks.
- **The tray lift and a load cell,** for a higher hand-off and a sensed take.
- **Venue retraining of the detector** (P5), and more objects (sock, remote, glasses). The dataset tooling is in #53.
- **MediaPipe hand height,** to choose a low or high present pose for a seated or standing person.
- **A sturdier drive:**
  - stronger motors than the 28BYJ-48 (capped at 30 mm/s tonight);
  - wheel encoders;
  - an E-stop sense wire;
  - a replacement for the wrist-roll servo, which is held fixed tonight.
- **A take sensor for the suction cup.**

---

## Not verified when this was written (01:30–02:30)

Nothing in this runbook was run on hardware. The commands come from the code on `integration/demo` and `task/HP-software` (#60). Their `--help` output and the simulated runs were checked; the hardware runs were not.

**Will fail until fixed:**

- **The `realsense` environment lacks PyYAML and Ultralytics.** `perception.publish` stops at `import yaml`, and `brain.check --role perception` reports `No module named 'yaml'`.
  - To fix: install them from `perception/requirements.txt`, then check that a RealSense frame still comes through (the S4 check).
  - Whether `--device mps` works in that environment is unknown.
- **Missing on SW1's Mac:**
  - the weights `data/models/phone_v2.pt` (on SW2's Mac);
  - `config/floor_homography.json`.
- **SW1's checkout is on `task/A13-teleop`.** Its `robot.yaml` has no `navigation` section and still has the old base values.
- **PlatformIO isn't installed on SW1's Mac:** flash the R4 from SW2's Mac, or install PlatformIO first.

**Unknown:**

- the R4's port name;
- whether the E-stop also cuts the steppers' 5 V;
- whether the phone resolves the `.local` link;
- whether the server needs a restart after every E-stop (assumed yes).

**Estimates:** the shot timings come from the configured speeds (approach 20 mm/s, search 15 mm/s, arm 25°/s), not from measurements.
