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
