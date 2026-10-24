"""Publish section 9.2 detections; --fake uses synthetic frames and detections."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import time

import numpy as np

from perception.realsense import FakeCamera, RealSense, Settings, ROOT


@dataclass(frozen=True)
class Detection:
    label: str
    conf: float
    bbox: tuple[float, float, float, float]


class YoloDetector:
    def __init__(self, weights, conf=0.5, device=None):
        # Fail locally rather than letting Ultralytics download a missing model.
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Missing detector weights: {weights}")
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf, self.device = conf, device

    def __call__(self, colour):
        result = self.model.predict(colour, imgsz=1280, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        return [Detection(result.names[int(cls)], float(conf), tuple(box))
                for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                          result.boxes.conf.cpu().tolist(),
                                          result.boxes.cls.cpu().tolist())]


class FakeDetector:
    """A clearly simulated phone box; no model imports or weights required."""
    def __call__(self, colour):
        h, w = colour.shape[:2]
        return [Detection("phone", 0.9, (w * .45, h * .6, w * .55, h * .7))]


def load_homography(path, settings):
    """P4 file: matrix, colour_size [w,h], flip; matrix maps pixels to metres."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    matrix = np.asarray(data["matrix"], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) != 3):
        raise ValueError("homography must be a finite, invertible 3x3 matrix")
    if tuple(data["colour_size"]) != tuple(settings.colour) or data["flip"] != settings.flip:
        raise ValueError("homography camera size/flip differs; redo P4 calibration")
    return matrix


def make_message(frame, detections, *, conf=0.5, homography=None):
    h, w = frame.colour.shape[:2]
    items = []
    for detection in detections:
        if (not math.isfinite(detection.conf) or not conf <= detection.conf <= 1
                or len(detection.bbox) != 4 or not np.isfinite(detection.bbox).all()):
            continue
        x1, y1, x2, y2 = detection.bbox
        x1, x2 = np.clip([x1, x2], 0, w).tolist()
        y1, y2 = np.clip([y1, y2], 0, h).tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        depth_m = None
        if (frame.aligned and frame.depth.shape == (h, w)
                and math.isfinite(frame.depth_scale) and frame.depth_scale > 0):
            region = frame.depth[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
            valid = region[np.isfinite(region) & (region > 0)]
            if valid.size:
                depth_m = float(np.median(valid)) * frame.depth_scale
        floor_m = None
        if homography is not None:
            point = homography @ np.array([cx, y2, 1.0])
            if np.isfinite(point).all() and abs(point[2]) > 1e-12:
                floor_m = (point[:2] / point[2]).tolist()
        items.append(dict(label=detection.label, conf=float(detection.conf),
                          bbox_px=[x1, y1, x2, y2], centre_px=[cx, cy],
                          floor_m=floor_m, depth_m=depth_m))
    return dict(t=frame.t, frame=frame.number, items=items)


def address(value):
    host, port = value.rsplit(":", 1)
    port = int(port)
    if not host or not 1 <= port <= 65535:
        raise ValueError("UDP address must be host:port (1–65535)")
    return host, port


def publish(camera, detector, sock, destination, *, conf=.5, homography=None,
            hz=15, count=None):
    sent = 0
    while count is None or sent < count:
        start = time.monotonic()
        frame = camera.read()
        message = make_message(frame, detector(frame.colour), conf=conf, homography=homography)
        sock.sendto(json.dumps(message, allow_nan=False).encode(), destination)
        sent += 1
        time.sleep(max(0, 1 / hz - (time.monotonic() - start)))
    return sent


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.yaml")
    parser.add_argument("--fake", action="store_true", help="no camera, model or hardware; synthetic phone")
    parser.add_argument("--count", type=int, help="exit after N messages")
    parser.add_argument("--hz", type=float, default=15)
    parser.add_argument("--device", help="YOLO device, e.g. mps or cpu")
    args = parser.parse_args(argv)
    if not math.isfinite(args.hz) or args.hz <= 0 or (args.count is not None and args.count <= 0):
        parser.error("hz and count must be positive")
    cfg = yaml.safe_load(args.config.read_text())["perception"]
    conf = float(cfg.get("detector_conf", .5))
    if not 0 <= conf <= 1:
        parser.error("detector_conf must be between 0 and 1")
    settings = Settings.load(args.config)
    weights = Path(cfg.get("detector_weights", "data/models/phone_v2.pt"))
    if not weights.is_absolute():
        weights = ROOT / weights
    detector = FakeDetector() if args.fake else YoloDetector(weights, conf, args.device)
    homography = None if args.fake else load_homography(args.config.parent / "floor_homography.json", settings)
    destination = address(cfg["detections_udp"])
    print(f"{'SIMULATED' if args.fake else 'LIVE'} detections → {destination}; floor mapping={homography is not None}", flush=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            with (FakeCamera if args.fake else RealSense)(settings) as camera:
                publish(camera, detector, sock, destination, conf=conf,
                        homography=homography, hz=args.hz, count=args.count)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
