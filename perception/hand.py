"""Hand height from the mast-top webcam (task L3 v2, plan 9.3 / 10.6). Env: perception.

    python -m perception.hand calibrate     # once per camera mount
    python -m perception.hand run           # publish the hand message, ~10 Hz
    python -m perception.hand run --fake    # synthetic hand, no camera (for brain tests)

Sends {"t", "hand_visible", "height_m", "conf"} as JSON to robot.yaml
perception.hand_udp (127.0.0.1:5602). MediaPipe's HandLandmarker finds the palm;
its pixel row becomes a height in metres through a straight-line fit made by
`calibrate` (open palm held at tape-measured heights where the person will reach
for the tray, so it holds at that distance from the mast).

The model file is downloaded once by the person (no network at run time):
    curl -L -o data/models/hand_landmarker.task <MODEL_URL below>
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"
CALIB_PATH = REPO_ROOT / "config" / "hand_calibration.json"
MODEL_PATH = REPO_ROOT / "data" / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
RATE_HZ = 10
PALM_LANDMARKS = (0, 5, 9, 17)      # wrist and the index / middle / pinky knuckles


def palm_row(landmarks, image_height: int) -> float:
    """Pixel row of the palm centre from MediaPipe's normalised hand landmarks."""
    return float(np.mean([landmarks[i].y for i in PALM_LANDMARKS]) * image_height)


class Calibration:
    """Straight-line fit: palm pixel row -> height above the floor (m)."""

    def __init__(self, rows_px, heights_m):
        if len(rows_px) < 2 or len(rows_px) != len(heights_m):
            raise ValueError("need at least 2 (row, height) pairs")
        if np.ptp(rows_px) < 20:
            raise ValueError("calibration rows are too close together; spread the heights out")
        self.rows_px = [float(r) for r in rows_px]
        self.heights_m = [float(h) for h in heights_m]
        self.slope, self.offset = np.polyfit(self.rows_px, self.heights_m, 1)

    def height(self, row_px: float) -> float:
        return float(self.slope * row_px + self.offset)

    def save(self, path: Path = CALIB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows_px": self.rows_px, "heights_m": self.heights_m}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(data["rows_px"], data["heights_m"])


def message(row_px: float | None, conf: float, calib: Calibration | None, t: float | None = None) -> dict:
    """The hand message (plan 9.3). height_m is None when no hand is seen."""
    visible = row_px is not None
    return {"t": round(t if t is not None else time.time(), 3), "hand_visible": visible,
            "height_m": round(calib.height(row_px), 3) if visible and calib else None,
            "conf": round(float(conf), 3) if visible else 0.0}


class MediaPipeHands:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            raise SystemExit(f"hand model missing; download it once:\n  mkdir -p {Path(model_path).parent} && "
                             f"curl -L -o {model_path} {MODEL_URL}")
        import mediapipe as mp

        self.mp = mp
        self.detector = mp.tasks.vision.HandLandmarker.create_from_model_path(str(model_path))

    def detect(self, rgb: np.ndarray):
        """Returns (palm_row_px, confidence) for the most confident hand, or None."""
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None
        scores = [h[0].score if h else 0.0 for h in result.handedness]
        best = int(np.argmax(scores))
        return palm_row(result.hand_landmarks[best], rgb.shape[0]), scores[best]


class Webcam:
    def __init__(self, index: int):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"can't open camera {index} (robot.yaml cameras.mast_top); is another program using it?")

    def read(self) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None

    def close(self) -> None:
        self.cap.release()


def _config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def _udp_target(cfg: dict) -> tuple[str, int]:
    host, port = cfg.get("perception", {}).get("hand_udp", "127.0.0.1:5602").rsplit(":", 1)
    return host, int(port)


def publish(read_frame, detect, calib, target, rate_hz=RATE_HZ, count=None, clock=time) -> None:
    """Read, detect, send; forever or for `count` messages."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / rate_hz
    sent = 0
    try:
        while count is None or sent < count:
            t0 = clock.monotonic()
            frame = read_frame()
            found = detect(frame) if frame is not None else None
            msg = message(found[0] if found else None, found[1] if found else 0.0, calib)
            sock.sendto(json.dumps(msg).encode(), target)
            sent += 1
            clock.sleep(max(0.0, period - (clock.monotonic() - t0)))
    finally:
        sock.close()


class FakeHand:
    """A hand drifting between 0.6 and 0.9 m, visible 90% of the time."""

    def __init__(self):
        self.calib = Calibration([0, 100], [1.0, 0.0])   # row = (1 - height) * 100
        self.t0 = time.monotonic()

    def read(self):
        return np.zeros((1, 1, 3), np.uint8)

    def detect(self, frame):
        t = time.monotonic() - self.t0
        if t % 10 > 9:
            return None
        height = 0.75 + 0.15 * math.sin(t / 3)
        return (1 - height) * 100, 0.9


def calibrate(cam, hands, ask=input) -> Calibration:
    print("Hold your OPEN palm flat, facing down, where you'd reach for the tray.\n"
          "At each height: measure from the floor to your palm, type it in mm, press Enter.\n"
          "Do at least 3 heights across the range (e.g. 600, 800, 1000). Empty line to finish.")
    rows, heights = [], []
    while True:
        text = ask(f"height in mm (#{len(rows) + 1}, empty to finish): ").strip()
        if not text:
            break
        try:
            mm = float(text)
        except ValueError:
            print("  type a number of mm")
            continue
        samples = [found[0] for found in (hands.detect(f) for f in (cam.read() for _ in range(10)) if f is not None)
                   if found]
        if len(samples) < 5:
            print("  hand not seen clearly; try again")
            continue
        rows.append(float(np.median(samples)))
        heights.append(mm / 1000)
        print(f"  {mm:.0f} mm -> palm at row {rows[-1]:.0f}")
    if len(rows) < 3:
        raise SystemExit("need at least 3 heights; nothing saved")
    calib = Calibration(rows, heights)
    worst = max(abs(calib.height(r) - h) for r, h in zip(rows, heights))
    print(f"fit error up to {worst * 100:.1f} cm")
    return calib


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m perception.hand")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--fake", action="store_true", help="synthetic hand, no camera")
    sub.add_parser("calibrate")
    args = p.parse_args(argv)

    cfg = _config()
    if args.cmd == "run" and args.fake:
        fake = FakeHand()
        publish(fake.read, fake.detect, fake.calib, _udp_target(cfg))
        return 0
    cam = Webcam(cfg.get("cameras", {}).get("mast_top", 3))
    try:
        hands = MediaPipeHands()
        if args.cmd == "calibrate":
            calibrate(cam, hands).save()
            print(f"saved {CALIB_PATH}")
        else:
            if not CALIB_PATH.exists():
                raise SystemExit(f"no {CALIB_PATH}; run `python -m perception.hand calibrate` first")
            publish(cam.read, hands.detect, Calibration.load(), _udp_target(cfg))
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
