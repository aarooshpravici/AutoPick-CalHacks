"""ACT grasp policy for pick.py (task T3, plan 10.2).

    policy = pick.load_policy("data/checkpoints/020000/pretrained_model")   # opens robot.yaml webcams
    pick_to_tray(arm, poses, policy=policy)

Loads a LeRobot checkpoint (weights plus its saved pre/post-processors) onto the
Mac's GPU (mps), and turns each arm state plus camera frames into joint targets,
the step(state) interface pick.py drives. Camera names must match the ones the
demos were recorded with (robot.yaml `cameras`: wrist, scene).

Cameras are captured the way training/record.py (task T1) records them: LeRobot's
OpenCV camera, RGB, at the checkpoint's image size (the dataset's, 320x180 by
default) and pick.RATE_HZ fps, which must equal the recording fps (T1's default
--fps 30); pick.py also steps the policy at that rate. A webcam may instead be
captured at a larger size with the same aspect ratio (as T1 does when the webcam
can't deliver 320x180; pick.py --camera-size), and frames are then resized to the
checkpoint's size. A frame with a different aspect ratio is an error: squashing
it would show ACT images unlike its demos.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .poses import JOINTS

IMAGE_PREFIX = "observation.images."
FRAME_MAX_AGE_MS = 200      # a webcam frame older than this means the camera stalled: error
ASPECT_TOLERANCE = 0.01     # relative difference in width/height still counted as the same shape


def checkpoint_image_sizes(path) -> dict[str, tuple[int, int]]:
    """{camera name: (width, height)} of each image the checkpoint takes, from its config.json."""
    with open(Path(path) / "config.json") as f:
        features = json.load(f)["input_features"]
    sizes = {}
    for key, feature in features.items():
        if key.startswith(IMAGE_PREFIX):
            _, height, width = feature["shape"]                     # (C, H, W)
            sizes[key.removeprefix(IMAGE_PREFIX)] = (int(width), int(height))
    return sizes


def same_aspect(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if (width, height) sizes a and b have the same aspect ratio."""
    ra, rb = a[0] / a[1], b[0] / b[1]
    return abs(ra - rb) <= ASPECT_TOLERANCE * rb


def default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActGrasp:
    def __init__(self, path, cameras: dict, device: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device or default_device())
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = str(path)
        cfg.device = self.device.type
        if hasattr(cfg, "pretrained_backbone_weights"):
            # The checkpoint's weights replace the ImageNet backbone anyway; without this,
            # torchvision would download ResNet weights at load time if its cache is empty.
            cfg.pretrained_backbone_weights = None
        self.config = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(str(path), config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": self.device.type}})

        self.image_sizes = {k.removeprefix(IMAGE_PREFIX): (int(f.shape[2]), int(f.shape[1]))   # (W, H)
                            for k, f in cfg.input_features.items() if k.startswith(IMAGE_PREFIX)}
        missing = set(self.image_sizes) - set(cameras)
        if missing:
            raise ValueError(f"checkpoint needs cameras {sorted(missing)}; have {sorted(cameras)}")
        self.cameras = cameras
        if tuple(cfg.output_features["action"].shape) != (len(JOINTS),):
            raise ValueError(f"checkpoint action shape {cfg.output_features['action'].shape}, expected (6,)")

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _frame(self, name: str) -> np.ndarray:
        frame = self.cameras[name].read()                       # H x W x 3, RGB, uint8
        width, height = self.image_sizes[name]
        got = (frame.shape[1], frame.shape[0])
        if got != (width, height):
            if not same_aspect(got, (width, height)):
                raise ValueError(f"camera {name!r} gives {got[0]}x{got[1]} frames but the checkpoint was "
                                 f"trained on {width}x{height}; resizing would squash them. Capture at "
                                 f"{width}x{height} or a size with the same aspect ratio")
            import cv2

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    def step(self, state: dict) -> dict:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {"observation.state": np.array([state[j] for j in JOINTS], dtype=np.float32)}
        for name in self.image_sizes:
            obs[IMAGE_PREFIX + name] = self._frame(name)
        with self.torch.inference_mode():
            batch = self.pre(prepare_observation_for_inference(obs, self.device))
            action = self.post(self.policy.select_action(batch))
        action = action.squeeze(0).to("cpu")
        return {j: float(action[i]) for i, j in enumerate(JOINTS)}


class OpenCVFrames:
    """A webcam by OpenCV index (robot.yaml `cameras`), giving RGB frames. Opened with the
    same LeRobot camera settings training/record.py passes to lerobot-record (index,
    width, height, fps), so the policy sees what was recorded. read() returns the latest
    frame without waiting and raises TimeoutError if it is older than FRAME_MAX_AGE_MS."""

    def __init__(self, index, width: int, height: int, fps: int):
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=fps, width=width, height=height))
        self.cam.connect()          # cleans up after itself if it fails

    def read(self) -> np.ndarray:
        return self.cam.read_latest(max_age_ms=FRAME_MAX_AGE_MS)

    def close(self) -> None:
        self.cam.disconnect()


class BlankFrames:
    """Grey frames of a given size, for dry runs with a real checkpoint."""

    def __init__(self, width: int, height: int):
        self.frame = np.full((height, width, 3), 128, dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self.frame

    def close(self) -> None:
        pass
