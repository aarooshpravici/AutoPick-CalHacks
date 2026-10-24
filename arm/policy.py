"""ACT grasp policy for pick.py (task T3, plan 10.2).

    policy = ActGrasp("data/checkpoints/020000/pretrained_model", cameras={"wrist": cam})
    pick_to_tray(arm, poses, policy=policy)

Loads a LeRobot checkpoint (weights plus its saved pre/post-processors) onto the
Mac's GPU (mps), and turns each arm state plus camera frames into joint targets,
the step(state) interface pick.py drives. Camera names must match the ones the
demos were recorded with (robot.yaml `cameras`: wrist, scene).
"""

from __future__ import annotations

import numpy as np

from .poses import JOINTS

IMAGE_PREFIX = "observation.images."


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
        self.policy = get_policy_class(cfg.type).from_pretrained(str(path), config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": self.device.type}})

        self.image_shapes = {k.removeprefix(IMAGE_PREFIX): tuple(f.shape)          # (C, H, W)
                             for k, f in cfg.input_features.items() if k.startswith(IMAGE_PREFIX)}
        missing = set(self.image_shapes) - set(cameras)
        if missing:
            raise ValueError(f"checkpoint needs cameras {sorted(missing)}; have {sorted(cameras)}")
        self.cameras = cameras
        if cfg.output_features["action"].shape != (len(JOINTS),):
            raise ValueError(f"checkpoint action shape {cfg.output_features['action'].shape}, expected (6,)")

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _frame(self, name: str) -> np.ndarray:
        import cv2

        frame = self.cameras[name].read()                       # H x W x 3, RGB, uint8
        _, h, w = self.image_shapes[name]
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    def step(self, state: dict) -> dict:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {"observation.state": np.array([state[j] for j in JOINTS], dtype=np.float32)}
        for name in self.image_shapes:
            obs[IMAGE_PREFIX + name] = self._frame(name)
        with self.torch.inference_mode():
            batch = self.pre(prepare_observation_for_inference(obs, self.device))
            action = self.post(self.policy.select_action(batch))
        action = action.squeeze(0).to("cpu")
        return {j: float(action[i]) for i, j in enumerate(JOINTS)}


class OpenCVFrames:
    """A webcam by OpenCV index (robot.yaml `cameras`), giving RGB frames."""

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        self.cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=fps, width=width, height=height))
        self.cam.connect()

    def read(self) -> np.ndarray:
        return self.cam.read()

    def close(self) -> None:
        self.cam.disconnect()


class BlankFrames:
    """Grey frames, for dry runs with a real checkpoint."""

    def __init__(self, width: int = 320, height: int = 180):
        self.frame = np.full((height, width, 3), 128, dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self.frame

    def close(self) -> None:
        pass
