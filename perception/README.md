# Perception environment (S4)

Use Python 3.10, independently of the robot's LeRobot 0.6.2 / Python 3.12
environment. On macOS, pyrealsense2-mac 2.54.1 supplies a Python 3.10 wheel
for macOS 13+ on Apple silicon and Intel:
https://pypi.org/project/pyrealsense2-mac/2.54.1/

From the repository root:

```sh
/opt/homebrew/bin/python3.10 -m venv .venv/perception
.venv/perception/bin/python -m pip install -r perception/requirements.txt
.venv/perception/bin/python -m pip check
.venv/perception/bin/python -c 'import sys, pyrealsense2, cv2, ultralytics, mediapipe, yaml; assert sys.version_info[:2] == (3, 10); print("perception imports OK")'
```

On Linux, substitute your `python3.10` executable. The requirements select the
Linux RealSense distribution automatically. These are direct dependency pins;
record the installed transitive versions with `pip freeze` under `data/` for
the robot Mac after validation. Never install these into the LeRobot environment.

## Local validation and known upstream limitation

Installed on SW2's Apple silicon Mac with Python 3.10.22. Imports of RealSense,
OpenCV, Ultralytics, MediaPipe and PyYAML pass. No camera was connected.
`pip check` currently exits nonzero with
`mediapipe 0.10.21 is not supported on this platform`: the universal2 wheel
contains an Intel-only tag in its internal WHEEL metadata. The actual package
imports on ARM64. MediaPipe 0.10.18 was checked and has the same defect, so the
original 0.10.21 pin is retained. Do not patch installed package metadata to
hide this result. Upstream report:
https://github.com/google-ai-edge/mediapipe/issues/6112

This is a known validation failure, not a clean `pip check` result. Recheck on
SW1's robot Mac, and keep this limitation visible until a compatible upstream
wheel is selected and tested. Camera capture remains a separate acceptance gate.

## S4 hardware acceptance (person runs this)

Plug the D435 into its own USB 3 port and close other camera programs. Run this
in your own terminal from the repository root; type your password there. The
agent must not run sudo. This opens only the camera; it does not move hardware.

```sh
sudo .venv/perception/bin/python - <<'PY'
import time
import pyrealsense2 as rs

for attempt in range(5):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    started = False
    try:
        pipe.start(cfg)
        started = True
        frames = pipe.wait_for_frames(5000)
        colour = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not colour or not depth:
            raise RuntimeError("Missing colour or depth frame")
        print("S4 frame captured:", colour.get_width(), colour.get_height(),
              depth.get_width(), depth.get_height())
        break
    except RuntimeError:
        if attempt == 4:
            raise
        time.sleep(1)
    finally:
        if started:
            pipe.stop()
PY
```

Report the result on issue #1. S4 is **not complete** until this captures a real
frame on the robot Mac. P1 subsequently supplies the reusable reader, alignment,
flip, snapshot, preview, and performance checks. Do not treat this smoke check
as P1's 25+ fps acceptance test.

## P3 phone detector and publisher

The trained checkpoint is already present on SW2's Mac. Copy it to the stable
runtime location; both paths stay ignored by git. Do not retrain it for this step.

```sh
mkdir -p data/models
cp -n data/runs/phone_v2/weights/best.pt data/models/phone_v2.pt
```

`perception.detector_weights` resolves relative to the repository root (or may
be absolute); `perception.detector_conf` defaults to 0.5. Inference uses imgsz
1280. The publisher targets 15 Hz, but actual throughput depends on inference;
the listener measures received messages. This model only identifies `phone`.

No-hardware smoke check (no model imports, no weights and no SDK):

```sh
.venv/perception/bin/python -m perception.publish --fake --count 30
```

Fake mode sends a synthetic phone box with synthetic depth and null floor
coordinates. Never use fake output as detector or parking acceptance evidence.

For the live check, leave the D435 fixed at the training mount height/tilt and
on its own USB 3 port. Close other camera programs and any process listening on
5601 (including the brain). This reader selects the RealSense SDK device, never
an OpenCV index or the built-in Mac camera. **The person runs this command** in
terminal A from the repository root and types their password:

```sh
sudo .venv/perception/bin/python -m perception.publish --device mps
```

To see the colour camera while detecting, stop the running publisher with
Ctrl-C, then restart it with `--preview`:

```sh
sudo .venv/perception/bin/python -m perception.publish --device mps --preview
```

The window shows the same boxes sent over UDP, with class, confidence and pixel
height. Click the window and press Q or Esc to quit (Ctrl-C in the terminal also
works). It uses the publisher's existing camera stream, so do not start a second
camera program. Preview also works with `--fake` without hardware or sudo.

If startup reports `UVC device is already opened`, close other camera apps and
capture scripts, then unplug/replug the D435 on its own USB 3 port and retry
one publisher. The reader now cleans up failed startup attempts and creates a
fresh pipeline for each retry. Each original startup error is printed, so if
it still fails, include the first error as well as the final traceback.
`floor mapping=False` is expected before P4 and does not prevent video.

In terminal B, place the phone and run a five-second labelled measurement:

```sh
.venv/perception/bin/python -m perception.listen --placement 0.3m-left
```

Repeat with labels `0.3m-centre`, `0.3m-right`, `0.5m-left`, `0.5m-centre`,
`0.5m-right`, `0.75m-left`, `0.75m-centre`, `0.75m-right`, `1.0m-left`,
`1.0m-centre`, `1.0m-right`, then `empty` (phone removed). Keep it still during
each measurement. Press Ctrl-C in terminal A when finished. The listener prints
messages and a SUMMARY with detection-frame count, maximum confidence, mean
pixel height and received Hz. Raw messages and summaries append to
`data/logs/p3-live.jsonl`; placement labels preserve misses and empty-floor false
positives too. Report all placements (at least 90% detected), no empty-floor
detections and >=10 Hz; do not select only favourable trials. Same-session
validation metrics are not this live acceptance check.

Depth is the median of positive finite aligned depth samples in the clipped
box, in metres, or null when unavailable. `floor_m` stays null until P4 supplies
`config/floor_homography.json` alongside the selected robot config:

```json
{"matrix": [[1,0,0],[0,1,0],[0,0,1]], "colour_size": [1280,720], "flip": false}
```

The example matrix is **only a schema example**, not a calibration. P4 must
replace it with a measured pixel-to-metre matrix: x forward, y left from the
floor below the camera. The box bottom-centre is mapped. Size/flip mismatch
fails startup; moving the camera requires recalibration even at the same size.
If distant phones fail, report the placement table and request additional far
phone/venue/empty-floor data using plan 10.3's fallback ladder; do not retrain
automatically.
