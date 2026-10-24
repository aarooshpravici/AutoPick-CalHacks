"""Hand height from the mast-top webcam (task L3, plan 9.3 / 10.6). Env: perception (Python 3.10).

    python -m perception.hand calibrate             # once per camera mount
    python -m perception.hand run                   # publish the hand message, ~10 Hz
    python -m perception.hand run --fake            # synthetic hand over UDP, no camera (brain tests)
    python -m perception.hand run --dry-run         # synthetic hand printed to stdout: no camera, no UDP
    python -m perception.hand calibrate --dry-run   # check robot.yaml and the files, open nothing

Every setting comes from config/robot.yaml (or --config), with no built-in
fallback: the camera index (cameras.mast_top), the UDP target
(perception.hand_udp, 127.0.0.1:5602) and the tray limits (lift.seated_hand_mm,
lift.standing_hand_mm). A missing or bad value is an error.

Sends {"t", "hand_visible", "height_m", "conf"} as JSON (plan 9.3). MediaPipe's
HandLandmarker finds the palm; its pixel row becomes a height in metres through a
straight-line fit made by `calibrate` (open palm held at tape-measured heights
where the person will reach for the tray, so it holds at that distance from the
mast). height_m is the hand's height, not clamped; it is null when no hand is seen.

Tray height (plan 10.6): `tray_target_mm` gives the brain the hand-off height in
the same frame as robot.yaml lift.*_hand_mm. v1 is the preset picked on the status
page ("seated" or "standing"). v2 is the hand's height from the latest hand
message, clamped between the two presets, and it falls back to the picked preset
when no hand is seen.

The model file is downloaded once by the person (no network at run time):
    curl -L -o data/models/hand_landmarker.task <MODEL_URL below>
"""

from __future__ import annotations

import argparse
import contextlib
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
RATE_HZ = 10                        # plan 9.3: about 10 Hz
DRY_RUN_COUNT = 10                  # messages `run --dry-run` prints unless --count is given
PALM_LANDMARKS = (0, 5, 9, 17)      # wrist and the index / middle / pinky knuckles
PRESETS = ("seated", "standing")    # status-page names for lift.seated_hand_mm / lift.standing_hand_mm


# --- robot.yaml -----------------------------------------------------------------

class ConfigError(ValueError):
    """robot.yaml is missing a value, or a value is unusable."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found (task S6 adds config/robot.yaml); pass --config to use another file")
    import yaml

    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} is empty or not a YAML mapping")
    return cfg


def _require(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            raise ConfigError(f"robot.yaml is missing {dotted}")
        node = node[part]
    return node


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def camera_index(cfg: dict) -> int:
    """OpenCV index of the mast-top webcam (robot.yaml cameras.mast_top)."""
    index = _require(cfg, "cameras.mast_top")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ConfigError(f"robot.yaml cameras.mast_top must be a camera index (0, 1, 2, ...), got {index!r}")
    return index


def udp_target(cfg: dict) -> tuple[str, int]:
    """(host, port) for the hand message (robot.yaml perception.hand_udp)."""
    value = _require(cfg, "perception.hand_udp")
    host, sep, port = str(value).rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ConfigError(f"robot.yaml perception.hand_udp must be host:port, got {value!r}")
    return host, int(port)


def presets_mm(cfg: dict) -> dict:
    """The v1 tray heights, which are also the v2 limits: {"seated": mm, "standing": mm}."""
    values = {}
    for name in PRESETS:
        key = f"lift.{name}_hand_mm"
        value = _require(cfg, key)
        if not _is_number(value):
            raise ConfigError(f"robot.yaml {key} must be a number of mm, got {value!r}")
        values[name] = float(value)
    if not 0 < values["seated"] < values["standing"]:
        raise ConfigError("robot.yaml needs 0 < lift.seated_hand_mm < lift.standing_hand_mm, "
                          f"got {values['seated']:g} and {values['standing']:g}")
    return values


def tray_target_mm(cfg: dict, preset: str, hand: dict | None = None) -> float:
    """Hand-off tray height (plan 10.6), in the frame of robot.yaml lift.*_hand_mm.

    v1: `preset` ("seated" or "standing", picked on the status page).
    v2: `hand` is the latest hand message (plan 9.3); the hand's height is used,
    clamped to [seated, standing]. No message, no visible hand or no height means
    the preset is used.
    """
    limits = presets_mm(cfg)
    if preset not in limits:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}, got {preset!r}")
    height_m = hand.get("height_m") if hand and hand.get("hand_visible") is True else None
    if not _is_number(height_m):
        return limits[preset]
    return min(max(height_m * 1000.0, limits["seated"]), limits["standing"])


# --- hand height ------------------------------------------------------------------

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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows_px": self.rows_px, "heights_m": self.heights_m}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(data["rows_px"], data["heights_m"])


def message(row_px: float | None, conf: float, calib: Calibration, t: float | None = None) -> dict:
    """The hand message (plan 9.3). height_m is None when no hand is seen."""
    visible = row_px is not None
    if visible and calib is None:
        raise ValueError("a visible hand needs a calibration; it must never be sent without a height")
    return {"t": round(t if t is not None else time.time(), 3), "hand_visible": visible,
            "height_m": round(calib.height(row_px), 3) if visible else None,
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

    def close(self) -> None:
        self.detector.close()


class Webcam:
    def __init__(self, index: int):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise SystemExit(f"can't open camera {index} (robot.yaml cameras.mast_top); is another program using it?")

    def read(self) -> np.ndarray | None:
        ok, bgr = self.cap.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None

    def close(self) -> None:
        self.cap.release()


# --- publishing -------------------------------------------------------------------

class UdpSender:
    """Sends each message as one JSON datagram to (host, port)."""

    def __init__(self, target: tuple[str, int]):
        self.target = target
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, msg: dict) -> None:
        self.sock.sendto(json.dumps(msg).encode(), self.target)

    def close(self) -> None:
        self.sock.close()


def print_message(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


def publish(read_frame, detect, calib, send, rate_hz=RATE_HZ, count=None, clock=time) -> None:
    """Read, detect, send(message); forever or for `count` messages."""
    period = 1.0 / rate_hz
    sent = 0
    while count is None or sent < count:
        t0 = clock.monotonic()
        frame = read_frame()
        found = detect(frame) if frame is not None else None
        send(message(found[0] if found else None, found[1] if found else 0.0, calib))
        sent += 1
        if count is None or sent < count:
            clock.sleep(max(0.0, period - (clock.monotonic() - t0)))


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


def calibrate(cam, hands, ask=None) -> Calibration:
    ask = ask or input
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
    try:
        calib = Calibration(rows, heights)
    except ValueError as e:
        raise SystemExit(f"{e}; nothing saved") from None
    worst = max(abs(calib.height(r) - h) for r, h in zip(rows, heights))
    print(f"fit error up to {worst * 100:.1f} cm")
    return calib


# --- command line -----------------------------------------------------------------

def _dry_run_report(args, cfg: dict, index: int, target: tuple[str, int]) -> None:
    limits = presets_mm(cfg)
    lines = [
        "dry run: no camera, no MediaPipe, no UDP.",
        f"camera: {index} (robot.yaml cameras.mast_top)",
        f"hand message: UDP {target[0]}:{target[1]} (robot.yaml perception.hand_udp), about {RATE_HZ} Hz",
        f"tray limits: seated {limits['seated']:g} mm, standing {limits['standing']:g} mm (robot.yaml lift)",
        f"calibration: {args.calibration} ({'found' if Path(args.calibration).exists() else 'missing; run calibrate'})",
        f"hand model: {args.model} " + ("(found)" if Path(args.model).exists()
                                        else f"(missing; download once: curl -L -o {args.model} {MODEL_URL})"),
    ]
    print("\n".join(lines), file=sys.stderr, flush=True)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=CONFIG_PATH, help="robot.yaml (default: config/robot.yaml)")
    common.add_argument("--calibration", type=Path, default=CALIB_PATH,
                        help="hand calibration file (default: config/hand_calibration.json)")
    common.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="MediaPipe hand_landmarker.task (default: data/models/hand_landmarker.task)")
    common.add_argument("--dry-run", action="store_true",
                        help="open no camera and no socket; run prints synthetic messages to stdout")

    p = argparse.ArgumentParser(prog="python -m perception.hand", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", parents=[common], help="publish the hand message (plan 9.3)")
    run.add_argument("--fake", action="store_true", help="synthetic hand, no camera (still sends UDP)")
    run.add_argument("--count", type=_positive_int, help=f"stop after N messages (dry run default: {DRY_RUN_COUNT})")
    sub.add_parser("calibrate", parents=[common], help="fit palm row -> height, once per camera mount")
    args = p.parse_args(argv)

    try:
        cfg = load_config(args.config)
        index, target = camera_index(cfg), udp_target(cfg)
        if args.dry_run:
            _dry_run_report(args, cfg, index, target)
    except ConfigError as e:
        raise SystemExit(f"error: {e}") from None
    count = getattr(args, "count", None)

    if args.dry_run:
        if args.cmd == "run":
            fake = FakeHand()
            publish(fake.read, fake.detect, fake.calib, print_message, count=count or DRY_RUN_COUNT)
        return 0

    with contextlib.ExitStack() as stack:
        if args.cmd == "run" and args.fake:
            fake = FakeHand()
            sender = UdpSender(target)
            stack.callback(sender.close)
            publish(fake.read, fake.detect, fake.calib, sender, count=count)
            return 0
        calib = None
        if args.cmd == "run":           # check every file before the camera opens
            if not Path(args.calibration).exists():
                raise SystemExit(f"no {args.calibration}; run `python -m perception.hand calibrate` first")
            calib = Calibration.load(args.calibration)
        hands = MediaPipeHands(args.model)
        stack.callback(hands.close)
        cam = Webcam(index)
        stack.callback(cam.close)
        if args.cmd == "calibrate":
            calibrate(cam, hands).save(args.calibration)
            print(f"saved {args.calibration}")
        else:
            sender = UdpSender(target)
            stack.callback(sender.close)
            publish(cam.read, hands.detect, calib, sender, count=count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
