"""Record ACT grasp demos with lerobot-record (task T1, plan 10.2).

    python -m training.record --name rehearsal --episodes 5            # prints the command
    python -m training.record --name grasp_v1 --episodes 50 [--scene] [--speed 25] --run

Builds the lerobot-record command from config/robot.yaml: the SO-101 follower and
leader, and the webcams under the fixed names ACT will see (`wrist`, and `scene`
with --scene). Datasets go to data/datasets/<name> (git-ignored), never the Hub.

Nothing moves by default: without --run the command is only printed. With --run
the person must confirm at the terminal that the area is clear (press Enter)
before lerobot-record starts; it enables torque and the follower starts copying
the leader straight away.

Speed: robot.yaml arm.move_speed_deg_s is the ceiling (--speed may lower it, never
raise it). lerobot-record has no speed setting, so the cap is applied as LeRobot's
per-command step limit: max_relative_target = min(max_relative_target_deg,
speed / fps). The same step applies to the gripper (0-100 scale), so at 25 deg/s
the claw takes about 4 s to close fully. The dataset stores the LEADER's positions,
so move the leader slowly enough for the follower to keep up.

Each demo starts at `ready` and ends the moment the claw is firmly closed on the
object: press the RIGHT ARROW key then to end the episode early (LEFT ARROW
re-records it, ESC stops). The scripted lift does the rest, so don't record it.

Cameras: LeRobot's OpenCV camera does not resize. If a webcam can't deliver
exactly 320x180 at the chosen fps it fails at connect. Check this on the real wrist
webcam in the T1 rehearsal; if it fails, record at a native size (for example
--width 640 --height 360) and resize to 320x180 in the convert step:
python -m training.convert --src data/datasets/NAME --dst data/datasets/NAME_320.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"
DATASETS = REPO_ROOT / "data" / "datasets"
TASK = "Pick up the object from the floor"
WIDTH, HEIGHT = 320, 180    # what ACT trains on (plan 10.2)
FPS = 30                    # must equal arm/pick.py RATE_HZ, the rate the policy runs at
EXE_NAME = "lerobot-record"

INSTRUCTIONS = """\
Recording {episodes} demos into {root}
  - Before you press Enter: put the LEADER arm at `ready`. As soon as lerobot-record
    starts, the follower moves to match the leader (at most {speed:g} deg/s, i.e.
    {step:.3f} per command at {fps} fps), and it then copies every leader move.
  - Move the LEADER arm slowly; the follower lags if you go faster than {speed:g} deg/s.
  - Close the claw firmly on the object, then press RIGHT ARROW at once to end the demo.
  - Between demos ({reset}s): move the object to a new spot/angle inside the parking window.
  - LEFT ARROW re-records the last demo. ESC stops and saves.
  - When lerobot-record stops, the follower goes limp: rest it or hold it before ESC.
  - Don't move any camera from now on; if one moves, the demos must be re-recorded.
"""


def load_config(path: Path) -> dict:
    import yaml

    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found; config/robot.yaml comes from task S6 (or pass --config)") from None
    except yaml.YAMLError as e:
        raise SystemExit(f"{path} is not valid YAML: {e}") from None
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} is empty or not a YAML mapping")
    return cfg


def need(cfg: dict, section: str, key: str):
    """cfg[section][key], or a clear error naming the missing robot.yaml key. No defaults."""
    block = cfg.get(section)
    if not isinstance(block, dict) or key not in block or block[key] is None:
        raise SystemExit(f"robot.yaml is missing `{section}.{key}`")
    return block[key]


def positive(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < value < math.inf):
        raise SystemExit(f"{label} must be a positive number, got {value!r}")
    return float(value)


def step_limit(cfg: dict, speed_deg_s: float | None, fps: int) -> tuple[float, float]:
    """(speed, per-command step) for LeRobot's max_relative_target, both capped by robot.yaml."""
    ceiling = positive(need(cfg, "arm", "move_speed_deg_s"), "robot.yaml arm.move_speed_deg_s")
    max_step = positive(need(cfg, "arm", "max_relative_target_deg"), "robot.yaml arm.max_relative_target_deg")
    speed = ceiling if speed_deg_s is None else positive(speed_deg_s, "--speed")
    if speed > ceiling:
        raise SystemExit(f"--speed {speed:g} is above robot.yaml arm.move_speed_deg_s ({ceiling:g}); refusing")
    return speed, min(max_step, speed / positive(fps, "--fps"))


def camera_spec(names, indices: dict, width: int, height: int, fps: int) -> str:
    missing = [n for n in names if n not in indices]
    if missing:
        raise SystemExit(f"robot.yaml `cameras` has no index for {missing}")
    parts = [f"{n}: {{type: opencv, index_or_path: {indices[n]}, width: {width}, height: {height}, fps: {fps}}}"
             for n in names]
    return "{ " + ", ".join(parts) + " }"


def build_command(cfg: dict, name: str, episodes: int, cameras: list[str], width: int, height: int,
                  fps: int, episode_time_s: int, reset_time_s: int, resume: bool, exe: str,
                  speed_deg_s: float | None = None) -> list[str]:
    _, step = step_limit(cfg, speed_deg_s, fps)
    indices = cfg.get("cameras")
    if not isinstance(indices, dict):
        raise SystemExit("robot.yaml has no `cameras` section")
    cmd = [
        exe,
        "--robot.type=so101_follower",
        f"--robot.port={need(cfg, 'arm', 'follower_port')}",
        f"--robot.id={need(cfg, 'arm', 'follower_id')}",
        f"--robot.max_relative_target={step:.3f}",
        f"--robot.cameras={camera_spec(cameras, indices, width, height, fps)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={need(cfg, 'arm', 'leader_port')}",
        f"--teleop.id={need(cfg, 'arm', 'leader_id')}",
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


def find_lerobot_record() -> str | None:
    """Full path of lerobot-record: next to this interpreter first, then on PATH."""
    return shutil.which(EXE_NAME, path=str(Path(sys.executable).parent)) or shutil.which(EXE_NAME)


def interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def confirm_area_clear() -> None:
    """Only a bare Enter from a person at a real terminal starts recording."""
    if not interactive():
        raise SystemExit("--run needs your person at an interactive terminal; nothing was started")
    try:
        answer = input("Area around the arm clear? Press Enter to start recording (anything else cancels): ")
    except (EOFError, KeyboardInterrupt):
        answer = None
    if answer is None or answer.strip():
        raise SystemExit("cancelled; nothing was started")


def _count(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value
    return parse


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.record",
                                description="Prints the lerobot-record command; --run records live.")
    p.add_argument("--name", required=True, help="dataset name, e.g. rehearsal or grasp_v1")
    p.add_argument("--episodes", type=_count(1), default=50)
    p.add_argument("--scene", action="store_true", help="also record the deck webcam as `scene` (decision D4)")
    p.add_argument("--width", type=_count(1), default=WIDTH)
    p.add_argument("--height", type=_count(1), default=HEIGHT)
    p.add_argument("--fps", type=_count(1), default=FPS)
    p.add_argument("--episode-time", type=_count(1), default=20, help="max seconds per demo")
    p.add_argument("--reset-time", type=_count(0), default=10, help="seconds to move the object between demos")
    p.add_argument("--speed", type=float, default=None,
                   help="follower speed cap in deg/s (default and maximum: robot.yaml arm.move_speed_deg_s)")
    p.add_argument("--resume", action="store_true", help="add demos to an existing dataset")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true",
                      help="record live: asks the person to confirm the area is clear, then starts lerobot-record")
    mode.add_argument("--print-only", "--dry-run", dest="print_only", action="store_true",
                      help="print the command, don't run it (the default)")
    args = p.parse_args(argv)

    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name may only use letters, digits, _ and -")
    root = DATASETS / args.name
    cfg = load_config(args.config)
    speed, step = step_limit(cfg, args.speed, args.fps)
    exe = find_lerobot_record()
    cmd = build_command(cfg, args.name, args.episodes, ["wrist", "scene"] if args.scene else ["wrist"],
                        args.width, args.height, args.fps, args.episode_time, args.reset_time, args.resume,
                        exe or EXE_NAME, speed)

    if not args.run:
        print(shlex.join(cmd))
        print("(printed only; add --run to record live)", file=sys.stderr)
        return 0

    if args.resume and not (root / "meta" / "info.json").exists():
        raise SystemExit(f"nothing to resume at {root} (no meta/info.json); check --name")
    if root.exists() and not args.resume:
        raise SystemExit(f"{root} already exists; pick a new --name or pass --resume")
    if exe is None:
        raise SystemExit(f"{EXE_NAME} not found next to {sys.executable} or on PATH; activate the lerobot env")
    print(INSTRUCTIONS.format(episodes=args.episodes, root=root, reset=args.reset_time,
                              speed=speed, step=step, fps=args.fps))
    confirm_area_clear()
    DATASETS.mkdir(parents=True, exist_ok=True)
    os.execv(exe, cmd)
    return 0  # not reached: execv replaces this process


if __name__ == "__main__":
    sys.exit(main())
