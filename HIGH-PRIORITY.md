# HIGH PRIORITY: demo-night context for SW2's Codex

*Written Sun 2026-10-25 01:45 by SW1's Claude (SW1 = @aarooshpravici). Goal: **a 5-minute demo video of AutoPick by 08:15 today.***

**Who does what tonight:**
- **SW2 (Juan, @romoj311-hub) owns this list.** SW2 makes the high-level calls, including what merges to main and in what order. SW1 handed that to SW2 at 01:30.
- **SW1** is on the arm (teleop) and helping HW1 (Darren) and HW2 (ClassicDQ) get the base rolling and steering.
- **SW1's Claude** answers arm questions on the board.

Read `plan.md` and `AGENTS.md` too. This file overrides the schedule in plan 14.6, which has passed.

## Rules that still hold (safety, from SW1)

1. **Never call the follower's servo 5 (wrist_roll).** It's faulty: it drives one way only and browns out the whole bus.
   - It's frozen in software: `config/robot.yaml` `arm.frozen_joints: [wrist_roll]`. `arm/real.py` `RealArm(frozen=...)` builds the bus without it.
   - Anything that builds LeRobot's `SO101Follower` directly and connects it (lerobot-teleoperate, lerobot-record, the old `safe_teleop.py`) talks to servo 5. **Don't use those.**
   - Never re-zero or recalibrate the follower's wrist roll. Never run a full `lerobot-calibrate` without SW1.
2. **Every teleop-style start shows the live leader/follower line-up table** (`arm/align.py` `live_align`). It starts only on Enter once the joints match.
3. **Nothing moves without the person.** A person runs every command that moves the arm or the wheels, and presses Enter. Agents never run real-mode hardware commands themselves.
   - **sudo** (perception) is typed by the person.
   - Before anything moves: area clear, hand near the E-stop.
4. **STOP must work in every state:** the page, the word "stop", and the E-stop. Never weaken a speed cap, the watchdog, or STOP to make the demo work.
5. **Tests:** check pytest's exit status. Don't pipe it through tail.
6. **SW1's Mac is the robot computer.** Use its Python environments, not `.venv`:
   - `/opt/miniconda3/envs/lerobot/bin/python`: arm, brain, ui and base;
   - `/opt/miniconda3/envs/realsense/bin/python`: RealSense perception, with sudo. It has no ultralytics or mediapipe yet; check before relying on it.
   - The repo is at `~/Desktop/Cal Hacks/AutoPick-CalHacks`.

## Hardware state (as of 01:30)

- **Follower arm:**
  - port `/dev/tty.usbmodem58CD1767761`, id `so101_follower_arm`, 12 V;
  - wrist roll frozen at its current orientation;
  - bus healthy (`~/Desktop/Cal Hacks/bus_health.py`: 0 failures).
- **Leader arm:**
  - port `/dev/tty.usbmodem58760435221`, id **`so101_leader_arm_b`**, 5 V;
  - its power plug has been loose once: if it shows 0 motors, reseat it.
- **Poses:** `config/poses.json` was taught **on the bench** (untracked, on SW1's Mac). It must be **re-taught on the base**, at the parking spot (H4).
- **Grasp:**
  - The claw can't lift a flat phone: the fingertips push it away.
  - A bunched sock or a TV remote works as the object.
  - An Octobuddy suction holder sticks to phones (it depends on the case); mounting it on the gripper is an open question (`--grasp suction`, PR #54).
- **Base:** Uno R4 WiFi with 28BYJ-48 steppers, USB to SW1's Mac. SW2 reports the wiring is finished. **It has never been driven by software.**
- **Arm reach:** about 13 in (33 cm). Never start a pick from farther away (REQ-claude-14, accepted).

## Open PRs (nothing below is on main yet)

**SW1:**
- #44: teach re-asks bad claw steps.
- #54: C5 suction mode, stacked on #44.
- #46: native demo recorder.
- #58: safe teleop `arm.teleop`, stacked on #44.
- #49: plan notes.

**SW2, the stack (in order):**
1. #32 P3
2. #48 P4 (and #53 P2/P3)
3. #34 B3
4. #50 B5
5. #51 B6
6. #52 M1-real
7. #56 UI
8. #57 web → R4

**Docs:** #30, #31, #55.

**Coming from SW1's background agents, by about 03:00** (watch issue #1):
- **`integration/demo`:** a branch where every open PR is already merged in order, with the full test suite and a simulated Phone mission plus STOP run on it. Use it to merge for real in minutes, or merge it directly if SW2 prefers.
- **A safety review of the stack:** firmware/serial, navigation, web → serial, and changes to `arm/`. Each finding is checked by a skeptic, and SW1's Claude will post the confirmed ones on issue #1.
- **`task/DEMO-runbook`:** `docs/demo-runbook.md`, with the pre-flight checks, the exact start-up commands, the shot list and the cover-ups.

## The tasks, in order

| # | Task | Done when | Target |
|---|---|---|---|
| H1 | ~~Safe teleop tool~~ **Done by SW1's Claude:** `python -m arm.teleop --speed 25` (#58) | — | done |
| H2 | **Merge to main.** Decide your own REQ-codex-6 (floor_homography.json schema), -7 (navigation settings in robot.yaml), -8 and -9 (web → R4, firmware/driver/brain). Use the review findings and `integration/demo`. Fix any confirmed blocker first. | main has every PR the demo needs; the full suite passes on main (exit 0) | 02:45 |
| H3 | **Base bring-up.** Flash the R4 from main. `python -m base.link` check (PING/STAT/STOP only, nothing moves). Then, with the wheels off the ground: forward, back, turn left, turn right, each confirmed to go **the right way**. Then on the floor. The page STOP and the E-stop stop it every time. | Directions right; STOP and E-stop verified on the floor | 03:30 |
| H4 | **Arm on the base.** SW1 runs `bus_health.py`, re-teaches with `python -m arm.teach --speed 25` at the parking spot, then `python -m arm.pick run --speed 25` 3/3 with the demo object | 3/3 pick → present → take from the parking spot | 04:30 |
| H5 | **Grasp for the video (D1).** Claw on the sock or remote, or Octobuddy via `--grasp suction` (#54). Pick one. | Decision posted on #1 | 04:30 |
| H6 | **Perception live** on SW1's Mac: RealSense + YOLO publisher (P3, sudo). Floor calibration at 3 measured points, plus the parking spot (P4). | `floor_m` within about 2 cm at 3 points; detections at about 15 Hz | 05:30 |
| H7 | **End-to-end on the real robot (M2).** Tap Phone on the page, then: search/approach (33 cm guard), park, pick, present, take. | At least 1 clean run on camera, then as many as time allows | 07:00 |
| H8 | **Shoot the video** from `docs/demo-runbook.md`, with the cover-ups below | Video exported | 08:15 |

**If time runs out, drop items from the bottom up:**
- Keep STOP and a real pick-and-present on camera.
- H6 and H7 can shrink: start the robot within about 1 m of the object, facing it.

## Cover-ups that are fine for a 5-minute video

**Fine:**
- start the robot close to the object and facing it;
- use the sock or remote instead of the phone;
- tethered or wall power;
- cut and retake;
- if autonomy fails, a teleoperated take that is **labelled as teleop** in the video.

**Not fine:**
- faking STOP;
- passing teleop off as autonomous.

**Say these are future work:**
- the ACT learned grasp (recorder #46 is ready; training on the lab PC);
- the tray/lift;
- venue retraining;
- MediaPipe hand height.
