"""Record ACT grasp demos with lerobot-record (task T1, plan 10.2).

    python -m training.record --name rehearsal --episodes 5
    python -m training.record --name grasp_v1 --episodes 50 [--scene] [--print-only]

Builds the lerobot-record command from config/robot.yaml: the SO-101 follower and
leader, and the webcams under the fixed names ACT will see (`wrist`, and `scene`
with --scene). Datasets go to data/datasets/<name> (git-ignored), never the Hub.

Each demo starts at `ready` and ends the moment the claw is firmly closed on the
object: press the RIGHT ARROW key then to end the episode early (LEFT ARROW
re-records it, ESC stops). The scripted lift does the rest, so don't record it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"
DATASETS = REPO_ROOT / "data" / "datasets"
TASK = "Pick up the object from the floor"

INSTRUCTIONS = """\
Recording {episodes} demos into {root}
  - Move the LEADER arm; the follower copies it. Start every demo at the `ready` pose.
  - Close the claw firmly on the object, then press RIGHT ARROW at once to end the demo.
  - Between demos ({reset}s): move the object to a new spot/angle inside the parking window.
  - LEFT ARROW re-records the last demo. ESC stops and saves.
  - Don't move any camera from now on; if one moves, the demos must be re-recorded.
"""


def load_config(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def camera_spec(names, indices: dict, width: int, height: int, fps: int) -> str:
    missing = [n for n in names if n not in indices]
    if missing:
        raise SystemExit(f"robot.yaml `cameras` has no index for {missing}")
    parts = [f"{n}: {{type: opencv, index_or_path: {indices[n]}, width: {width}, height: {height}, fps: {fps}}}"
             for n in names]
    return "{ " + ", ".join(parts) + " }"


def build_command(cfg: dict, name: str, episodes: int, cameras: list[str], width: int, height: int,
                  fps: int, episode_time_s: int, reset_time_s: int, resume: bool, exe: str) -> list[str]:
    arm = cfg["arm"]
    cmd = [
        exe,
        "--robot.type=so101_follower",
        f"--robot.port={arm['follower_port']}",
        f"--robot.id={arm['follower_id']}",
        f"--robot.max_relative_target={arm.get('max_relative_target_deg', 10)}",
        f"--robot.cameras={camera_spec(cameras, cfg.get('cameras', {}), width, height, fps)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={arm['leader_port']}",
        f"--teleop.id={arm['leader_id']}",
        f"--dataset.repo_id=local/{name}",
        f"--dataset.root={DATASETS / name}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.single_task={TASK}",
        f"--dataset.fps={fps}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    ]
    if resume:
        cmd.append("--resume=true")
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.record")
    p.add_argument("--name", required=True, help="dataset name, e.g. rehearsal or grasp_v1")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--scene", action="store_true", help="also record the deck webcam as `scene` (decision D4)")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-time", type=int, default=20, help="max seconds per demo")
    p.add_argument("--reset-time", type=int, default=10, help="seconds to move the object between demos")
    p.add_argument("--resume", action="store_true", help="add demos to an existing dataset")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--print-only", action="store_true", help="print the command, don't run it")
    args = p.parse_args(argv)

    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name may only use letters, digits, _ and -")
    root = DATASETS / args.name
    if root.exists() and not args.resume and not args.print_only:
        raise SystemExit(f"{root} already exists; pick a new --name or pass --resume")
    exe = shutil.which("lerobot-record", path=str(Path(sys.executable).parent)) or "lerobot-record"
    cmd = build_command(load_config(args.config), args.name, args.episodes,
                        ["wrist", "scene"] if args.scene else ["wrist"], args.width, args.height,
                        args.fps, args.episode_time, args.reset_time, args.resume, exe)
    if args.print_only:
        print(" ".join(f"'{c}'" if " " in c else c for c in cmd))
        return 0
    print(INSTRUCTIONS.format(episodes=args.episodes, root=root, reset=args.reset_time))
    DATASETS.mkdir(parents=True, exist_ok=True)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
