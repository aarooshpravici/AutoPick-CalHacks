"""D435 colour/depth reader and no-hardware synthetic camera (P1)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def retry(call, *, sleep=time.sleep):
    """Five total attempts, one second apart; retain the original SDK error."""
    for attempt in range(5):
        try:
            return call()
        except RuntimeError:
            if attempt == 4:
                raise
            sleep(1)


@dataclass(frozen=True)
class Settings:
    colour: tuple[int, int]
    depth: tuple[int, int]
    fps: int
    align_depth_to_colour: bool
    flip: bool

    def __post_init__(self):
        for size in (self.colour, self.depth):
            if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                raise ValueError("stream sizes must be two positive integers")
        if type(self.fps) is not int or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if any(type(v) is not bool for v in (self.flip, self.align_depth_to_colour)):
            raise ValueError("flip and align_depth_to_colour must be booleans")

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as stream:
            cfg = yaml.safe_load(stream)["realsense"]
        return cls(**{name: cfg[name] for name in cls.__dataclass_fields__})


@dataclass
class Frame:
    colour: np.ndarray  # BGR uint8, owned memory; safe after SDK frame release.
    depth: np.ndarray   # uint16 sensor units; zero means invalid depth.
    depth_scale: float  # metres per sensor unit, from the device.
    t: float            # Unix seconds for the section 9.2 publisher.
    number: int
    aligned: bool


class RealSense:
    def __init__(self, settings: Settings, *, sdk=None, sleep=time.sleep):
        self.settings = settings
        self.sdk = sdk
        self.sleep = sleep
        self.pipeline = None
        self.aligner = None

    def _call(self, call):
        return retry(call, sleep=self.sleep)

    def __enter__(self):
        if self.pipeline is not None:
            raise RuntimeError("camera already open")
        if self.sdk is None:
            import pyrealsense2
            self.sdk = pyrealsense2
        rs = self.sdk
        pipe = self._call(rs.pipeline)
        cfg = self._call(rs.config)
        self._call(lambda: cfg.enable_stream(rs.stream.color, *self.settings.colour,
                                            rs.format.bgr8, self.settings.fps))
        self._call(lambda: cfg.enable_stream(rs.stream.depth, *self.settings.depth,
                                            rs.format.z16, self.settings.fps))
        self._call(lambda: pipe.start(cfg))
        self.pipeline = pipe
        try:
            profile = self._call(pipe.get_active_profile)
            device = self._call(profile.get_device)
            sensor = self._call(device.first_depth_sensor)
            self.depth_scale = self._call(sensor.get_depth_scale)
            if self.settings.align_depth_to_colour:
                self.aligner = self._call(lambda: rs.align(rs.stream.color))
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        if self.pipeline is None:
            raise RuntimeError("camera is not open")

        def acquire():
            frames = self.pipeline.wait_for_frames(5000)
            if self.aligner is not None:
                frames = self.aligner.process(frames)
            colour = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not colour or not depth:
                raise RuntimeError("missing colour or depth frame")
            return (np.asanyarray(colour.get_data()).copy(),
                    np.asanyarray(depth.get_data()).copy(), colour.get_frame_number())

        colour, depth, number = self._call(acquire)
        if self.settings.flip:
            colour, depth = colour[::-1, ::-1].copy(), depth[::-1, ::-1].copy()
        return Frame(colour, depth, self.depth_scale, time.time(), number,
                     self.settings.align_depth_to_colour)

    def close(self):
        if self.pipeline is not None:
            self._call(self.pipeline.stop)
            self.pipeline = None
            self.aligner = None

    def __exit__(self, *_):
        self.close()


class FakeCamera:
    """Deterministic asymmetric test frames; never imports the camera SDK."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.number = 0

    def __enter__(self):
        return self

    def read(self):
        self.number += 1
        w, h = self.settings.colour
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        colour = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
        dw, dh = self.settings.colour if self.settings.align_depth_to_colour else self.settings.depth
        depth = np.full((dh, dw), 1000, dtype=np.uint16)
        depth[0, 0] = 0
        if self.settings.flip:
            colour, depth = colour[::-1, ::-1].copy(), depth[::-1, ::-1].copy()
        return Frame(colour, depth, 0.001, time.time(), self.number,
                     self.settings.align_depth_to_colour)

    def __exit__(self, *_):
        pass


def return_ownership(path):
    """Only files/directories created by this process are passed here."""
    if os.geteuid() == 0 and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
        os.chown(path, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))


def make_output_dir(path):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        return_ownership(directory)


def snapshot(frame, directory, *, simulated=False):
    import cv2
    make_output_dir(directory)
    # Unique names avoid overwriting dataset photos across runs.
    stem = ("sim_" if simulated else "") + f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    colour = directory / f"{stem}.png"
    depth = directory / f"{stem}_depth.png"
    metadata = directory / f"{stem}.json"
    for path, data in ((colour, frame.colour), (depth, frame.depth)):
        if not cv2.imwrite(str(path), data):
            raise OSError(f"could not write {path}")
        return_ownership(path)
    metadata.write_text(json.dumps({"t": frame.t, "frame": frame.number,
                                   "depth_scale_m": frame.depth_scale,
                                   "aligned": frame.aligned, "simulated": simulated}) + "\n")
    return_ownership(metadata)
    return colour


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.yaml")
    parser.add_argument("--fake", action="store_true", help="synthetic camera; no hardware")
    sub = parser.add_subparsers(dest="command", required=True)
    shot = sub.add_parser("snapshot")
    shot.add_argument("--count", type=int, default=1)
    shot.add_argument("--every", type=float, default=1)
    shot.add_argument("--output", type=Path)
    sub.add_parser("preview")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--count", type=int, default=300)
    args = parser.parse_args(argv)
    if getattr(args, "count", 1) <= 0 or not np.isfinite(getattr(args, "every", 0)) or getattr(args, "every", 0) < 0:
        parser.error("count must be positive and every must be finite and nonnegative")
    settings = Settings.load(args.config)
    factory = FakeCamera if args.fake else RealSense
    with factory(settings) as camera:
        if args.command == "snapshot":
            output = args.output or ROOT / "data" / ("simulated" if args.fake else "photos")
            # Keep draining the stream between snapshots to avoid stale queued frames.
            saved, deadline = 0, time.monotonic()
            while saved < args.count:
                frame = camera.read()
                if time.monotonic() >= deadline:
                    print(snapshot(frame, output, simulated=args.fake), flush=True)
                    saved += 1
                    deadline = time.monotonic() + args.every
                if args.fake:
                    time.sleep(1 / settings.fps)
        elif args.command == "benchmark":
            for _ in range(30):
                camera.read()
            start = time.monotonic()
            for _ in range(args.count):
                camera.read()
            fps = args.count / (time.monotonic() - start)
            print(json.dumps({"fps": fps, "frames": args.count, "simulated": args.fake,
                              "aligned": settings.align_depth_to_colour}))
        else:
            import cv2
            try:
                while True:
                    frame = camera.read()
                    cv2.imshow("colour (q to quit)", frame.colour)
                    cv2.imshow("depth", cv2.convertScaleAbs(frame.depth, alpha=0.03))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if args.fake:
                        time.sleep(1 / settings.fps)
            finally:
                cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
