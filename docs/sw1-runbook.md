# SW1 runbook: Saturday from 09:00

Every command for the arm, demos, training and testing, in order. Run them from the repo folder in the `lerobot` environment. Plan references: 10.1 (arm), 10.2 (ACT), 14.6 (schedule).

## Every session, first

```bash
cd "$HOME/Desktop/Cal Hacks/AutoPick-CalHacks" && git pull && conda activate lerobot
```

```bash
python -m arm.check
```

`arm.check` reads only: it never moves anything. Fix every ✗ before going on.

- **Port ✗:** run `lerobot-find-port` once per arm and update `arm.follower_port` / `arm.leader_port` in `config/robot.yaml`. macOS renumbers ports after replugging.
- **wrist_flex warning:** see step 2.
- **Poses ✗:** expected until step 4.

**Safety, all day**

- **Before any motion:** clear the area, and keep a hand near the arm's power.
- **First real runs:** use `--speed 25`.
- **Stopping:** Ctrl-C in the terminal stops the arm and it holds its pose. You can also use the status page STOP or the E-stop.
- **Teleop:** rest the follower before stopping teleop, because it goes limp.
- **After the 50 demos are recorded, no camera may move.**

## 1. 09:00: on-the-base gate (G2)

The arm, the wrist camera and the low front camera (about 7–12 cm up, about 20° down) are on their final mounts. The arm's mount height is fixed from now on.

## 2. Recalibrate the leader (recommended, about 5 min)

The leader's wrist was calibrated over only 126° against the follower's 177°, so the follower's wrist can't use its full range. A backup of both calibration files is in `data/calibration/`.

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem58760435221 --teleop.id=so101_leader_arm
```

1. At the first Enter, hold every joint at the middle of its travel.
2. Sweep every joint through its whole range, including the wrist fully both ways and the folded rest pose.
3. Press Enter.

`python -m arm.check` should then show no wrist warning.

## 3. Teleop check

Put the leader in roughly the follower's pose first: the follower jumps to match it.

```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58CD1767761 --robot.id=so101_follower_arm --robot.max_relative_target=10 --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem58760435221 --teleop.id=so101_leader_arm
```

Check that each joint follows, and that a folded leader folds the follower to rest.

## 4. Re-teach the poses on the base (A6)

```bash
python -m arm.teach
```

Move the leader to each pose and press Enter:

| Pose | What it should look like |
|---|---|
| `ready` | Folded, out of the front camera's view |
| `above_object` | Claw open, just above the taped pickup spot |
| `at_object` | Claw open, fingertips at the floor around the object. For the phone, put the fixed jaw against the far edge (the backstop). |
| `lift` | Holding height for carrying |
| `present` | The highest comfortable reach, claw facing the person |
| claw open / closed | Claw fully open, then fully closed on nothing |

## 5. Dry run, then real runs

```bash
python -m arm.pick run --dry-run
```

```bash
python -m arm.pick run --speed 25
```

- **The take:** pull the object out only after the arm stops at `present`.
- **Targets:** 3/3 at 25 deg/s, then 3/3 with no `--speed` (45).
- **Hard to pull out?** Add `--handoff-grip light`, and if that's better set `handoff.grip: light` in `robot.yaml`.
- **Exit codes:** 0 taken, 2 missed, 3 dropped, 4 stuck, 5 timeout, 1 error or stopped.

## 6. 09:45: record the 50 demos (T4; SW2 parks the base)

`--run` records natively: it never calls the frozen wrist_roll servo, has no start-up jolt, and lines the arms up before the follower moves. Without `--run` it only prints the equivalent `lerobot-record` command; don't run that, because it would command servo 5.

```bash
python -m training.record --name grasp_v1 --episodes 50 --run
```

1. **Area clear?** Press Enter.
2. **Line-up table:** the follower holds still. Move the leader until every joint shows ✓, then press Enter.
3. **Before demo 1:** the follower now copies the leader, but nothing is recorded yet. Bring the leader to `ready`, place the object, and press Enter to start demo 1.
4. **Each demo:** starts at `ready` and **ends the moment the claw is firmly closed on the object**: press Enter then (20 s at most). The scripted moves do the rest.
5. **Reset (10 s):** the arm still follows, nothing is recorded. Move the object and bring the leader back to `ready`. Enter starts the next demo early.

Keys, typed in the recording terminal and followed by Enter:

- **`r`:** discards the demo in progress, or during its reset the one just finished, and records it again.
- **`q`:** saves the finished demos and stops. A demo still in progress is not saved, so press Enter first to keep it.
- **Ctrl-C:** stops now. The follower holds where it is; saved demos are kept and the unsaved one is dropped, including a finished demo still in its reset (`q` is the key that saves it). A demo that is being saved always finishes first, even if you press Ctrl-C again.
- **A save failed** (for example, a full disk)? The dataset is marked broken (`RECORDING_BROKEN.txt`) and `--resume` refuses it. Record the remaining demos into a new `--name`.

- **At the end:** the follower keeps holding its pose with the torque on. Rest it or hold it before you switch it off.
- **Stopped part-way?** Add the rest with `--resume`, for example `python -m training.record --name grasp_v1 --episodes 20 --run --resume`. Without `--resume`, an existing dataset is refused.
- **Too slow?** A demo recorded below 90% of 30 fps gets a warning; press `r` in its reset to drop it.
- **Placement:** vary the object about ±2 cm and a few angles across the parking window, so ACT learns to correct for parking.
- **Cameras:** record the wrist camera only. Leave out `--scene` unless D4 says otherwise: two-camera video encoding crashed once in testing.
- **If the wrist webcam can't capture 320×180:** record with `--width 640 --height 360`, then convert:

```bash
python -m training.convert --src data/datasets/grasp_v1 --dst data/datasets/grasp_v1_320
```

## 7. 11:30: train on the PC (T5)

The lab PC (RTX 5070 Ti, Ubuntu 22.04, shared account `keshav`) is reachable only from the university network, so turn on the university VPN at the venue. `remote.sh` uses its own folder, `~/floor-picker-train`, away from other projects. It checks the GPU is free and refuses to start over anyone's job. It never stores GitHub or Hugging Face tokens. Run these in a Terminal tab: Claude's sandboxed shell has no route to the PC.

```bash
export FP_TRAIN_HOST=keshav@10.193.169.55
```

Check it's reachable first. If this times out, the VPN is off or the PC is asleep.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 keshav@10.193.169.55 'nvidia-smi; command -v python3.12; git --version'
```

`setup` needs `python3.12` and `git` on the PC. Ubuntu 22.04 ships Python 3.10, so if `python3.12` is missing, tell Claude.

```bash
training/remote.sh setup
```

```bash
training/remote.sh run data/datasets/grasp_v1 grasp_v1
```

- **Progress:** `training/remote.sh status grasp_v1`
- **Copy checkpoints back:** `training/remote.sh pull grasp_v1`, as they appear (every 5,000 steps).
- **Stop:** `training/remote.sh stop grasp_v1`
- **Converted dataset:** if you converted, train on `data/datasets/grasp_v1_320` instead.

## 8. Test a checkpoint (T5)

First check it loads on this Mac: a simulated arm, but the real webcams and checkpoint. Nothing moves.

```bash
python -m arm.pick run --dry-run --policy data/checkpoints/grasp_v1/005000/pretrained_model
```

Then on the arm:

```bash
python -m arm.pick run --policy data/checkpoints/grasp_v1/020000/pretrained_model --speed 25
```

- **640×360 recording:** add `--camera-size 640x360` if you recorded at 640x360.
- **Decision:** at least 7 of 10 picks ending `taken` (phone placed randomly in the parking window) means ACT; otherwise stay with the scripted grasp.

## 9. Status page

```bash
python -m ui.server
```

- **Opening it:** http://127.0.0.1:8000. With no flags it runs a fake brain, which is safe for checking the page.
- **The real robot:** `--real`, once SW2's brain (M1) exists.

## What to tell Claude

- After step 1: any ✗ from `arm.check`.
- After step 5: the pick results, and squeeze vs light.
- After step 6: whether the wrist camera did 320×180 or needed converting.
- During step 7: the PC's hostname, and when checkpoints arrive.
