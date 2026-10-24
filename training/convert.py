"""Resize a recorded dataset's camera images: the convert step (task T2, plan 10.2).

    python -m training.convert --src data/datasets/grasp_v1 --dst data/datasets/grasp_v1_320
                               [--width 320 --height 180]

A webcam that can't deliver 320x180 is recorded at a native size with the same
aspect ratio (training/record.py --width 640 --height 360). This writes a new
LeRobot dataset with every camera image resized to --width x --height with
cv2.INTER_AREA, the same resize arm/policy.py applies to live frames, so ACT
trains on images like the ones it will see. Episodes, tasks, actions, state and
fps are copied unchanged; the image feature shapes and all stats are rewritten
by LeRobot's own dataset writer, and videos are re-encoded with the source's
encoder settings.

Refuses to change the aspect ratio (that would squash the images), never
overwrites --dst, and never writes to --src. Works offline: nothing is fetched
from the Hub. The new dataset is built in a hidden folder next to --dst and only
renamed to --dst once it is complete, so a failed run leaves no half-written --dst.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from arm.policy import same_aspect

from .record import HEIGHT, WIDTH

# LeRobot fills these in itself for every frame (add_frame / save_episode).
AUTO_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")
VISUAL = ("image", "video")


def read_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise SystemExit(f"{root} is not a LeRobot dataset (no meta/info.json)")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}") from None


def image_size(name: str, feature: dict) -> tuple[int, int]:
    """(width, height) of a camera feature stored as (height, width, channels), as LeRobot records it."""
    names = feature.get("names") or []
    if list(names[:2]) != ["height", "width"] or len(feature["shape"]) != 3:
        raise SystemExit(f"camera {name} has shape {feature['shape']} with names {names}; "
                         f"expected (height, width, channels)")
    height, width, _ = feature["shape"]
    return int(width), int(height)


def camera_sizes(features: dict, width: int, height: int) -> dict[str, tuple[int, int]]:
    """{camera key: (width, height)} of every camera, after checking each may become width x height."""
    if width < 1 or height < 1:
        raise SystemExit(f"--width and --height must be positive, got {width}x{height}")
    cams = {k: f for k, f in features.items() if f.get("dtype") in VISUAL}
    if not cams:
        raise SystemExit("the dataset has no camera images to resize")
    sizes = {}
    for key, feature in cams.items():
        if (feature.get("info") or {}).get("is_depth_map") or feature["shape"][-1] != 3:
            raise SystemExit(f"{key} is not an RGB camera; only RGB images can be converted")
        size = image_size(key, feature)
        if not same_aspect(size, (width, height)):
            raise SystemExit(f"{key} is {size[0]}x{size[1]}; {width}x{height} has a different aspect ratio "
                             f"and would squash the images. Pick a size with the same aspect ratio")
        if feature["dtype"] == "video" and (width % 2 or height % 2):
            raise SystemExit(f"{key} is stored as video, which needs an even width and height; "
                             f"got {width}x{height}")
        sizes[key] = size
    return sizes


def target_features(features: dict, width: int, height: int) -> dict:
    """The source's features minus the automatic ones, with each camera at width x height.
    A camera keeps only is_depth_map from its info: the writer fills in the video details."""
    out = {}
    for key, feature in features.items():
        if key in AUTO_FEATURES:
            continue
        feature = json.loads(json.dumps(feature))
        if feature.get("dtype") in VISUAL:
            feature["shape"] = [height, width, feature["shape"][2]]
            feature["info"] = {"is_depth_map": False}
        out[key] = feature
    return out


def to_hwc_uint8(image) -> np.ndarray:
    """A LeRobot frame (C, H, W tensor, uint8 or float in [0, 1]) as H x W x C uint8."""
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array.transpose(1, 2, 0))


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The resize arm/policy.py applies to live frames."""
    import cv2

    if (image.shape[1], image.shape[0]) == (width, height):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _frame(item: dict, features: dict, width: int, height: int) -> dict:
    frame = {"task": item["task"]}
    for key, feature in features.items():
        value = item[key]
        if feature["dtype"] in VISUAL:
            frame[key] = resize(to_hwc_uint8(value), width, height)
        elif feature["dtype"] == "string":
            frame[key] = value
        else:
            array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            frame[key] = array.astype(feature["dtype"]).reshape(feature["shape"])
    return frame


def _save(dataset, episode: int, frames: int, log) -> None:
    # One camera after another: LeRobot's default encodes each camera in its own process, and one
    # of those worker processes died (BrokenProcessPool) once in six two-camera rehearsals (rehearse_sim).
    dataset.save_episode(parallel_encoding=False)
    log(f"episode {episode}: {frames} frames")


def convert(src, dst, width: int = WIDTH, height: int = HEIGHT, log=print) -> dict:
    """Write dst: the dataset at src with every camera image resized to width x height.
    Returns {"episodes", "frames", "cameras": {key: ((w, h) before, (w, h) after)}}."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    info = read_info(src)
    if dst.exists():
        raise SystemExit(f"{dst} already exists; it is never overwritten. Pick a new --dst")
    if dst == src or src in dst.parents:
        raise SystemExit(f"--dst must be outside --src ({src})")
    sizes = camera_sizes(info["features"], width, height)
    features = target_features(info["features"], width, height)
    videos = [k for k, f in features.items() if f["dtype"] == "video"]

    from lerobot.configs.video import encoder_config_from_video_info
    from lerobot.datasets import LeRobotDataset

    source = LeRobotDataset(f"local/{src.name}", root=src, return_uint8=True)
    rgb_encoder = encoder_config_from_video_info(info["features"][videos[0]].get("info")) if videos else None
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent))
    try:
        out = LeRobotDataset.create(
            f"local/{dst.name}", info["fps"], features, root=work / "dataset", robot_type=info.get("robot_type"),
            use_videos=bool(videos), rgb_encoder=rgb_encoder, image_writer_threads=4 * len(sizes),
            data_files_size_in_mb=info.get("data_files_size_in_mb"),
            video_files_size_in_mb=info.get("video_files_size_in_mb"))
        try:
            episode, count = None, 0
            for i in range(len(source)):
                item = source[i]
                index = int(item["episode_index"])
                if episode is not None and index != episode:
                    _save(out, episode, count, log)
                    count = 0
                episode = index
                out.add_frame(_frame(item, features, width, height))
                count += 1
            if episode is not None:
                _save(out, episode, count, log)
        finally:
            out.finalize()
        if dst.exists():
            raise SystemExit(f"{dst} appeared while converting; left it alone")
        os.rename(work / "dataset", dst)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result = LeRobotDataset(f"local/{dst.name}", root=dst)
    if (result.meta.total_episodes, result.meta.total_frames) != (source.meta.total_episodes,
                                                                  source.meta.total_frames):
        raise SystemExit(f"{dst} has {result.meta.total_episodes} episodes / {result.meta.total_frames} frames; "
                         f"{src} has {source.meta.total_episodes} / {source.meta.total_frames}")
    return {"episodes": result.meta.total_episodes, "frames": result.meta.total_frames,
            "cameras": {key: (size, (width, height)) for key, size in sizes.items()}}


def _count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.convert",
                                description="Copy a LeRobot dataset with its camera images resized.")
    p.add_argument("--src", type=Path, required=True, help="recorded dataset, e.g. data/datasets/grasp_v1")
    p.add_argument("--dst", type=Path, required=True, help="new dataset folder (must not exist)")
    p.add_argument("--width", type=_count, default=WIDTH, help=f"default {WIDTH}, what ACT trains on")
    p.add_argument("--height", type=_count, default=HEIGHT, help=f"default {HEIGHT}")
    args = p.parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"      # local datasets only: never fall back to the Hub
    summary = convert(args.src, args.dst, args.width, args.height)
    for key, (before, after) in summary["cameras"].items():
        print(f"{key}: {before[0]}x{before[1]} -> {after[0]}x{after[1]}")
    print(f"wrote {args.dst}: {summary['episodes']} episodes, {summary['frames']} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
