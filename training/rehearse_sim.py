"""Rehearse the whole ACT pipeline in simulation (task T2, plan 10.2).

    python -m training.rehearse_sim [--steps 20] [--out DIR] [--episodes 3] [--scene] [--record-size 640x360]

training/record.py, training/remote.sh and arm/policy.py were written apart. This
runs them together, with no hardware and no network, and reports each stage:

  record   A LeRobot dataset with the features lerobot-record gives the command
           record.py builds (so101_follower, OpenCV cameras `wrist`, plus `scene`
           with --scene): the command is parsed with lerobot-record's own config
           parser, and the features come from the same LeRobot helpers lerobot-record
           applies to the robot's observation and action features. The robot is
           built from that config but never connected (no serial port, no camera).
           Frames come from the simulated arm (arm/sim.py) the way lerobot-record
           takes them (follower observation, then the leader's action): short demos
           ready -> at_object -> claw closed on the object, the object at another pan
           angle in each, with synthetic webcam images in which the object is a
           coloured rectangle that moves with the arm.
  convert  Only with --record-size other than record.py's 320x180: training/convert.py.
           The load stage then captures at the recorded size and resizes, as
           pick.py --camera-size does.
  train    training/remote.sh run, status and pull, with FP_TRAIN_HOST=local, so
           lerobot-train gets exactly remote.sh's flags. The "PC" is a folder under
           OUT whose venv/bin/lerobot-train runs this machine's lerobot-train with the
           changes a Mac needs: device mps (or cpu), batch 2, no dataloader workers,
           and no pretrained ResNet weights (nothing to download). Offline: no Hub,
           no wandb. remote.sh run gets --force because a Mac has no nvidia-smi.
  load     The pulled pretrained_model, first made to read like one from the PC
           (device cuda, LeRobot's default ImageNet backbone named), through arm/pick.py
           load_policy, the call pick_to_hand(..., policy=DIR) makes, with simulated
           webcams standing in for arm/policy.py OpenCVFrames (they render the same
           synthetic view from the sim arm) and any weight download refused. Checks the
           checkpoint's cameras, image sizes and state/action shapes against the
           dataset, and takes one policy step.
  run      One simulated pick_to_hand(arm, poses, policy=DIR), as the brain will call
           it, to a result. Any result is fine (the policy has barely trained); the
           point is that every interface lines up.

Everything goes under OUT (default: a new temporary folder): datasets, the stand-in
PC, pulled checkpoints and the Hugging Face caches. Exit code 0 when every stage ran.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from arm import pick
from arm import policy as policy_mod
from arm.poses import JOINTS
from arm.sim import DEFAULT_POSES, SimArm

from . import record

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SH = REPO_ROOT / "training" / "remote.sh"
NAME = "sim_rehearsal"
PC_ROOT = "floor-picker-train"                  # remote.sh's default FP_TRAIN_ROOT
TRAIN_TIMEOUT_S = 1800
POLL_S = 2.0

# The simulated robot.yaml (plan 9.1): only what record.py and pick.py read. The ports are
# never opened: the robot is built from its config to read its features, never connected.
SIM_ROBOT_YAML = {
    "arm": {"follower_port": "/dev/null-sim-follower", "leader_port": "/dev/null-sim-leader",
            "follower_id": "sim_follower", "leader_id": "sim_leader",
            "max_relative_target_deg": 10, "move_speed_deg_s": 45},
    "cameras": {"wrist": 0, "scene": 1},
    "handoff": {"timeout_s": 60},
}

# Demos: the leader moves at this speed, opens the claw on the way down, closes it on the
# object, and the demo ends HOLD_FRAMES after the claw is closed (plan 10.2).
DEMO_SPEED_DEG_S = 45.0
CLOSE_FRAMES = 15
HOLD_FRAMES = 10
OBJECT_PANS = (0.0, -8.0, 8.0, -4.0, 4.0)       # where the object is in each demo, deg of shoulder_pan

# Synthetic webcam images, in RGB like LeRobot's OpenCV camera.
FLOOR_RGB = (96, 96, 96)
OBJECT_RGB = (220, 40, 40)
JAW_RGB = (30, 30, 30)
CLAW_RGB = (40, 80, 220)
APPROACH_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


# --- synthetic images --------------------------------------------------------------

def approach(state: dict) -> float:
    """How far the arm has come from `ready` toward `at_object`: 0 at ready, 1 there."""
    start, goal = DEFAULT_POSES["ready"], DEFAULT_POSES["at_object"]
    d = np.array([goal[j] - start[j] for j in APPROACH_JOINTS])
    s = np.array([state[j] - start[j] for j in APPROACH_JOINTS])
    return float(np.clip(s @ d / (d @ d), 0.0, 1.0))


def _box(img: np.ndarray, cx: float, cy: float, half_w: float, half_h: float, rgb) -> None:
    """Fill a box given in fractions of the image size (so every resolution shows the same picture)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, round((cx - half_w) * w)), min(w, round((cx + half_w) * w))
    y0, y1 = max(0, round((cy - half_h) * h)), min(h, round((cy + half_h) * h))
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = rgb


def render(camera: str, state: dict, width: int, height: int, object_pan: float = 0.0) -> np.ndarray:
    """A synthetic webcam frame (height x width x 3, uint8, RGB) for the arm in `state`.
    wrist: the object is centred when the arm pans to it and grows as the claw comes down;
    the claw's jaws open and close at the bottom. scene: a fixed view of the floor, with
    the object and the claw seen from the deck."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = FLOOR_RGB
    p = approach(state)
    if camera == "wrist":
        size = 0.06 + 0.14 * p
        _box(img, 0.5 + 0.03 * (object_pan - state["shoulder_pan"]), 0.25 + 0.4 * p, size, size * 0.8, OBJECT_RGB)
        gap = 0.04 + 0.3 * float(np.clip(state["gripper"], 0, 100)) / 100
        for side in (-1, 1):
            _box(img, 0.5 + side * gap, 0.92, 0.03, 0.08, JAW_RGB)
    else:
        _box(img, 0.5 + 0.02 * object_pan, 0.75, 0.05, 0.04, OBJECT_RGB)
        _box(img, 0.5 + 0.02 * state["shoulder_pan"], 0.2 + 0.5 * p, 0.04, 0.04, CLAW_RGB)
    return img


# --- record ------------------------------------------------------------------------

def record_config(cameras: list[str], width: int, height: int, episodes: int = 3):
    """record.py's lerobot-record command for the simulated robot.yaml, parsed by lerobot-record's parser."""
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    cmd = record.build_command(SIM_ROBOT_YAML, NAME, episodes, cameras, width, height, record.FPS,
                               episode_time_s=20, reset_time_s=10, resume=False, exe=record.EXE_NAME)
    return draccus.parse(RecordConfig, args=cmd[1:])


def record_features(rec, calibration_dir: Path):
    """(robot, dataset features) exactly as lerobot-record computes them (lerobot_record.py, record()):
    LeRobot's feature helpers on the robot's action and observation features. The robot is built
    from rec.robot but never connected."""
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.feature_utils import combine_feature_dicts

    rec.robot.calibration_dir = Path(calibration_dir)  # not ~/.cache: nothing outside OUT
    robot = make_robot_from_config(rec.robot)
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=rec.dataset.video),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=rec.dataset.video),
    )
    return robot, features


def leader_path(object_pan: float, fps: int) -> list[dict]:
    """The leader's positions, one per frame: ready -> at_object (claw opening) -> claw closed -> hold."""
    poses = DEFAULT_POSES
    start = dict(poses["ready"])
    goal = {**poses["at_object"], "shoulder_pan": object_pan, "gripper": poses["gripper_open"]}
    n = max(1, math.ceil(max(abs(goal[j] - start[j]) for j in JOINTS) / (DEMO_SPEED_DEG_S / fps)))
    path = [{j: start[j] + (goal[j] - start[j]) * k / n for j in JOINTS} for k in range(1, n + 1)]
    for k in range(1, CLOSE_FRAMES + 1):
        g = goal["gripper"] + (poses["gripper_closed"] - goal["gripper"]) * k / CLOSE_FRAMES
        path.append({**goal, "gripper": g})
    path += [{**goal, "gripper": poses["gripper_closed"]}] * HOLD_FRAMES
    return path


def record_demo(dataset, features: dict, cameras: dict, task: str, object_pan: float, fps: int) -> int:
    """One demo into dataset's episode buffer, frame by frame as lerobot-record's record_loop does:
    read the follower, read the leader, send the leader's position to the follower, store both.
    cameras: {name: (width, height)}. Returns the number of frames; the claw must end on the object."""
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    arm = SimArm()
    path = leader_path(object_pan, fps)
    for target in path:
        state = arm.read()
        obs = {f"{j}.pos": state[j] for j in JOINTS}
        obs.update({name: render(name, state, w, h, object_pan) for name, (w, h) in cameras.items()})
        action = {f"{j}.pos": target[j] for j in JOINTS}
        arm.command(target)
        dataset.add_frame({**build_dataset_frame(features, obs, prefix=OBS_STR),
                           **build_dataset_frame(features, action, prefix=ACTION), "task": task})
        arm.clock.sleep(1.0 / fps)
    if arm.object != "held":
        raise RuntimeError(f"demo with the object at pan {object_pan:g} did not end holding it ({arm.object})")
    return len(path)


def record_stage(out: Path, cameras: list[str], size: tuple[int, int], episodes: int, log) -> tuple[Path, str]:
    from lerobot.datasets import LeRobotDataset

    rec = record_config(cameras, *size, episodes)
    robot, features = record_features(rec, out / "calibration")
    root = out / "datasets" / NAME              # record.py would use the repo's data/datasets/NAME
    rec.dataset.stamp_repo_id()
    dataset = LeRobotDataset.create(
        rec.dataset.repo_id, rec.dataset.fps, root=root, robot_type=robot.name, features=features,
        use_videos=rec.dataset.video,
        image_writer_threads=rec.dataset.num_image_writer_threads_per_camera * len(robot.cameras))
    sizes = {name: (cam.width, cam.height) for name, cam in rec.robot.cameras.items()}
    frames = 0
    try:
        for i in range(episodes):
            pan = OBJECT_PANS[i % len(OBJECT_PANS)]
            n = record_demo(dataset, features, sizes, rec.dataset.single_task, pan, rec.dataset.fps)
            # One camera after another: LeRobot's default encodes each camera in its own process, and
            # here one of those worker processes died once in six two-camera runs (BrokenProcessPool).
            dataset.save_episode(parallel_encoding=False)
            frames += n
            log(f"  demo {i}: object at pan {pan:+g} deg, {n} frames, claw closed on it")
    finally:
        dataset.finalize()
    cams = ", ".join(f"{n} {w}x{h}" for n, (w, h) in sizes.items())
    return root, (f"{episodes} demos, {frames} frames at {rec.dataset.fps} fps, robot_type {robot.name}, "
                  f"cameras {cams}; features {sorted(features)}")


# --- train -------------------------------------------------------------------------

SHIM = """#!/usr/bin/env bash
# Stand-in for the training PC's venv, written by training/rehearse_sim.py: this machine's
# lerobot-train, given remote.sh's arguments unchanged except for the Mac changes below.
printf '%s\\n' "$@" > "$HOME/lerobot-train.args"
args=()
for a in "$@"; do
  case "$a" in
    --policy.device=*) args+=("--policy.device={device}") ;;
    *) args+=("$a") ;;
  esac
done
exec {exe} "${{args[@]}}" {extra}
"""


def find_lerobot_train() -> str:
    exe = shutil.which("lerobot-train", path=str(Path(sys.executable).parent)) or shutil.which("lerobot-train")
    if exe is None:
        raise RuntimeError(f"lerobot-train not found next to {sys.executable} or on PATH; "
                           f"activate the lerobot env")
    return exe


def mac_overrides(steps: int) -> list[str]:
    """What the rehearsal adds to remote.sh's lerobot-train flags, all of them things remote.sh
    leaves at LeRobot's defaults: a tiny batch, no worker processes, no ImageNet download."""
    return ["--batch_size=2", "--num_workers=0", "--policy.pretrained_backbone_weights=null",
            f"--log_freq={max(1, steps // 4)}"]


def write_shim(path: Path, exe: str, device: str, steps: int) -> None:
    import shlex

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SHIM.format(device=device, exe=shlex.quote(exe), extra=shlex.join(mac_overrides(steps))))
    path.chmod(0o755)


def pc_env(out: Path) -> dict:
    """remote.sh's environment: the PC is this machine (FP_TRAIN_HOST=local) with its home under OUT."""
    home = out / "pc"
    cache = home / ".cache" / "huggingface"
    return dict(os.environ, HOME=str(home), FP_TRAIN_HOST="local", FP_TRAIN_ROOT=PC_ROOT,
                FP_CHECKPOINT_DIR=str(out / "checkpoints"), FP_START_WAIT="5",
                HF_HOME=str(cache), HF_LEROBOT_HOME=str(cache / "lerobot"), HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", TORCH_HOME=str(home / ".cache" / "torch"))


def remote(env: dict, *args: str, timeout: float = 300) -> str:
    r = subprocess.run(["bash", str(REMOTE_SH), *args], env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"remote.sh {' '.join(args)} failed ({r.returncode}):\n{r.stderr}{r.stdout}")
    return r.stdout


def train_stage(out: Path, dataset: Path, steps: int, device: str, log) -> tuple[Path, str]:
    env = pc_env(out)
    pc = Path(env["HOME"]) / PC_ROOT
    write_shim(pc / "venv" / "bin" / "lerobot-train", find_lerobot_train(), device, steps)
    t0 = time.monotonic()
    log("  " + remote(env, "run", str(dataset), NAME, "--steps", str(steps), "--force").strip())
    while True:
        status = remote(env, "status", NAME)
        if status.startswith("not running"):
            break
        if time.monotonic() - t0 > TRAIN_TIMEOUT_S:
            remote(env, "stop", NAME)
            raise RuntimeError(f"training did not finish within {TRAIN_TIMEOUT_S} s; stopped it")
        time.sleep(POLL_S)
    seconds = time.monotonic() - t0
    train_log = (pc / "logs" / f"{NAME}.log").read_text(errors="replace")
    if "End of training" not in train_log:
        tail = "\n".join(train_log.splitlines()[-25:])
        raise RuntimeError(f"lerobot-train did not finish training. Last lines of its log:\n{tail}")
    losses = [line.split("loss:")[1].split()[0] for line in train_log.splitlines() if " loss:" in line]
    log("  " + remote(env, "pull", NAME).strip().replace("\n", "\n  "))
    pulled = sorted((out / "checkpoints" / NAME).glob("*/pretrained_model"))
    if not pulled:
        raise RuntimeError(f"remote.sh pull brought back no pretrained_model into {out / 'checkpoints' / NAME}")
    flags = (Path(env["HOME"]) / "lerobot-train.args").read_text().splitlines()
    policy_flags = " ".join(a for a in flags if a.startswith("--policy."))
    return pulled[-1], (f"{steps} steps on {device} in {seconds:.0f} s via remote.sh run/status/pull "
                        f"({len(flags)} remote.sh flags incl. {policy_flags}; device -> {device}; "
                        f"added {' '.join(mac_overrides(steps))}); "
                        f"loss {losses[0] if losses else '?'} -> {losses[-1] if losses else '?'}; "
                        f"checkpoint {pulled[-1]}")


# --- load and run ------------------------------------------------------------------

@contextlib.contextmanager
def _swapped(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def sim_webcams(holder: dict, robot_yaml: dict, object_pan: float = 0.0):
    """A stand-in for arm/policy.py OpenCVFrames (same constructor: robot.yaml index, width,
    height, fps) that renders the synthetic view of the camera with that index from the sim
    arm in holder["arm"]. Every instance is listed in the class's `opened`."""
    names = {index: name for name, index in robot_yaml["cameras"].items()}

    class SimWebcam:
        opened = []

        def __init__(self, index, width: int, height: int, fps: int):
            self.name, self.size, self.fps, self.closed = names[index], (width, height), fps, False
            SimWebcam.opened.append(self)

        def read(self) -> np.ndarray:
            return render(self.name, holder["arm"].read(), *self.size, object_pan)

        def close(self) -> None:
            self.closed = True

    return SimWebcam


def _refuse_download(*args, **kwargs):
    raise RuntimeError("tried to download pretrained weights; the rehearsal is offline")


@contextlib.contextmanager
def sim_hardware(out: Path, holder: dict):
    """pick.py as on the robot Mac, except robot.yaml is the simulated one, the webcams are
    simulated, and fetching pretrained weights (torch hub, torchvision) is an error."""
    import torch
    import torchvision.models._api as tv_api
    import yaml

    path = out / "robot.yaml"
    path.write_text(yaml.safe_dump(SIM_ROBOT_YAML))
    webcam = sim_webcams(holder, SIM_ROBOT_YAML)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_swapped(policy_mod, "OpenCVFrames", webcam))
        stack.enter_context(_swapped(pick, "DEFAULT_CONFIG_PATH", path))
        stack.enter_context(_swapped(torch.hub, "load_state_dict_from_url", _refuse_download))
        stack.enter_context(_swapped(tv_api, "load_state_dict_from_url", _refuse_download))
        yield webcam


def as_trained_on_pc(checkpoint: Path) -> None:
    """Make a checkpoint trained here read like one remote.sh trains on the PC: device cuda, and
    LeRobot's default ImageNet ResNet named as the backbone (only this rehearsal turns it off, to
    avoid the download). arm/policy.py must load either on the Mac without fetching anything."""
    import json

    from lerobot.policies.act.configuration_act import ACTConfig

    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config.update(device="cuda", pretrained_backbone_weights=ACTConfig.pretrained_backbone_weights)
    path.write_text(json.dumps(config, indent=4))
    path = checkpoint / "policy_preprocessor.json"
    pipeline = json.loads(path.read_text())
    for step in pipeline["steps"]:
        if step["registry_name"] == "device_processor":
            step["config"]["device"] = "cuda"
    path.write_text(json.dumps(pipeline, indent=2))


def check_checkpoint(policy, dataset_root: Path) -> str:
    """The checkpoint's inputs and outputs against the dataset it was trained on."""
    import json

    features = json.loads((dataset_root / "meta" / "info.json").read_text())["features"]
    cams = {k.removeprefix(policy_mod.IMAGE_PREFIX): (f["shape"][1], f["shape"][0])
            for k, f in features.items() if k.startswith(policy_mod.IMAGE_PREFIX)}
    if policy.image_sizes != cams:
        raise RuntimeError(f"checkpoint cameras {policy.image_sizes} != dataset cameras {cams}")
    names = features["observation.state"]["names"]
    if names != [f"{j}.pos" for j in JOINTS] or features["action"]["names"] != names:
        raise RuntimeError(f"dataset joints {names} are not arm/poses.py JOINTS {JOINTS} in order")
    state = tuple(policy.config.input_features["observation.state"].shape)
    action = tuple(policy.config.output_features["action"].shape)
    if state != (len(JOINTS),) or action != (len(JOINTS),):
        raise RuntimeError(f"checkpoint state {state} / action {action}, expected ({len(JOINTS)},)")
    return ", ".join(f"{n} {w}x{h}" for n, (w, h) in cams.items())


def load_stage(out: Path, checkpoint: Path, dataset_root: Path,
               capture_size: tuple[int, int] | None = None) -> str:
    """capture_size: the size the demos were recorded at, when they were converted (pick.py --camera-size)."""
    as_trained_on_pc(checkpoint)
    holder = {"arm": SimArm()}
    with sim_hardware(out, holder) as webcam:
        policy = pick.load_policy(checkpoint, capture_size=capture_size)
        try:
            cams = check_checkpoint(policy, dataset_root)
            policy.reset()
            action = policy.step(holder["arm"].read())
        finally:
            notes = pick.close_cameras(policy.cameras)
    if notes or not all(c.closed for c in webcam.opened):
        raise RuntimeError(f"cameras not closed: {notes}")
    if set(action) != set(JOINTS) or not all(math.isfinite(v) for v in action.values()):
        raise RuntimeError(f"policy step gave {action}")
    opened = ", ".join(f"{c.name} {c.size[0]}x{c.size[1]} @ {c.fps} fps" for c in webcam.opened)
    resized = f" (capture size {capture_size[0]}x{capture_size[1]}, resized)" if capture_size else ""
    return (f"PC-style checkpoint (device cuda, ImageNet backbone named) -> ActGrasp on {policy.device.type}, "
            f"nothing downloaded; checkpoint cameras {cams} match the dataset; webcams opened "
            f"as {opened}{resized}; one step from ready -> " + ", ".join(f"{j} {action[j]:.1f}" for j in JOINTS))


def run_stage(out: Path, checkpoint: Path, log) -> str:
    holder = {"arm": SimArm()}
    with sim_hardware(out, holder) as webcam:
        t0 = time.monotonic()
        result = pick.pick_to_hand(holder["arm"], DEFAULT_POSES, policy=str(checkpoint), log=log)
        seconds = time.monotonic() - t0
    if not webcam.opened or not all(c.closed for c in webcam.opened):
        raise RuntimeError("pick_to_hand did not open and close its webcams")
    return (f"pick_to_hand -> {result['result']} (grasp {result['grasp']}, {result['attempts']} attempt(s), "
            f"{result['seconds']} s simulated, {seconds:.1f} s real; arm ended with the object "
            f"{holder['arm'].object}){': ' + result['detail'] if result['detail'] else ''}")


# --- the rehearsal -----------------------------------------------------------------

def rehearse(out: Path, steps: int = 20, episodes: int = 3, scene: bool = False,
             record_size: tuple[int, int] = (record.WIDTH, record.HEIGHT), device: str | None = None,
             log=print) -> list[tuple[str, str, str]]:
    """Run every stage in order; a failed stage stops the rest. Returns [(stage, outcome, detail)]."""
    from training import convert

    device = device or policy_mod.default_device()
    cameras = ["wrist", "scene"] if scene else ["wrist"]
    results = []
    state = {}

    def do_record():
        state["dataset"], detail = record_stage(out, cameras, record_size, episodes, log)
        return detail

    def do_convert():
        if not converted:
            return None
        dst = out / "datasets" / f"{NAME}_{record.WIDTH}"
        summary = convert.convert(state["dataset"], dst, record.WIDTH, record.HEIGHT, log=lambda m: log("  " + m))
        state["dataset"] = dst
        cams = ", ".join(f"{k.removeprefix(policy_mod.IMAGE_PREFIX)} {b[0]}x{b[1]} -> {a[0]}x{a[1]}"
                         for k, (b, a) in summary["cameras"].items())
        return f"{cams}; {summary['episodes']} episodes, {summary['frames']} frames kept"

    def do_train():
        state["checkpoint"], detail = train_stage(out, state["dataset"], steps, device, log)
        return detail

    converted = record_size != (record.WIDTH, record.HEIGHT)
    stages = [("record", do_record), ("convert", do_convert), ("train", do_train),
              ("load", lambda: load_stage(out, state["checkpoint"], state["dataset"],
                                          record_size if converted else None)),
              ("run", lambda: run_stage(out, state["checkpoint"], log))]
    failed = False
    for name, stage in stages:
        if failed:
            results.append((name, "not run", "an earlier stage failed"))
            continue
        log(f"== {name}")
        t0 = time.monotonic()
        try:
            detail = stage()
        except Exception as e:  # noqa: BLE001 - report the stage and stop
            failed = True
            log(traceback.format_exc())
            results.append((name, "FAILED", f"{type(e).__name__}: {e}"))
            continue
        if detail is None:
            results.append((name, "skipped", f"recorded at {record.WIDTH}x{record.HEIGHT} already"))
        else:
            results.append((name, "ok", f"{detail} [{time.monotonic() - t0:.0f} s]"))
    return results


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 640x360, got {text!r}") from None
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"expected a positive size, got {text!r}")
    return w, h


def _count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.rehearse_sim",
                                description="Record, convert, train, load and run ACT in simulation.")
    p.add_argument("--steps", type=_count, default=20, help="training steps (default 20)")
    p.add_argument("--out", type=Path, help="work folder, new or empty (default: a new temporary folder)")
    p.add_argument("--episodes", type=_count, default=3, help="simulated demos to record (default 3)")
    p.add_argument("--scene", action="store_true", help="also record and use the `scene` camera")
    p.add_argument("--record-size", type=_size, default=(record.WIDTH, record.HEIGHT), metavar="WxH",
                   help=f"webcam size to record at (default {record.WIDTH}x{record.HEIGHT}); another size "
                        f"with the same aspect ratio adds the convert stage")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), help="training device (default: mps if available)")
    args = p.parse_args(argv)

    if args.out is None:
        out = Path(tempfile.mkdtemp(prefix="rehearse_sim_"))
    else:
        out = args.out.resolve()
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty; pick a new --out")
        out.mkdir(parents=True, exist_ok=True)
    # Before LeRobot is imported: offline, and every cache under OUT.
    os.environ.update(HF_HOME=str(out / "hf_home"), HF_LEROBOT_HOME=str(out / "hf_home" / "lerobot"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled")
    print(f"rehearsal folder: {out}")
    results = rehearse(out, args.steps, args.episodes, args.scene, args.record_size, args.device)
    print("\nstage     outcome  detail")
    for name, outcome, detail in results:
        print(f"{name:<9} {outcome:<8} {detail}")
    return 0 if all(outcome in ("ok", "skipped") for _, outcome, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
