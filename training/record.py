"""Record ACT grasp demos (task T1, plan 10.2).

    python -m training.record --name rehearsal --episodes 5            # prints the lerobot-record command
    python -m training.record --name grasp_v1 --episodes 50 [--scene] [--speed 25] --run
    python -m training.record --name grasp_v1 --episodes 10 --run --resume     # 10 more into grasp_v1

The recording is described as a lerobot-record command built from config/robot.yaml: the
SO-101 follower and leader, and the webcams under the fixed names ACT will see (`wrist`, and
`scene` with --scene). Datasets go to data/datasets/<name> (git-ignored), never the Hub.
Without --run the command is only printed and nothing moves.

--run records natively instead of running that command. lerobot-record builds the follower
itself, so it would command servo 5 (wrist_roll, faulty: when commanded it browns out the
whole bus; robot.yaml arm.frozen_joints), it jolts at start-up, and it has no line-up gate.
--run takes everything else from the same command, parsed by lerobot-record's own parser
(dataset name and folder, task, fps, demo and reset times, cameras), and writes the dataset
lerobot-record would:

  1. "Area clear?": only a bare Enter from a person at a real terminal goes on.
  2. Opens the leader, then the webcams (LeRobot's OpenCV camera, exactly as the command
     describes them), then the follower through arm/real.py RealArm: no frozen servo is ever
     called, and each goal is set to where the servo is before the torque ramps up (no jolt).
  3. The live leader/follower table (arm/align.py): the follower holds still until every
     joint matches and Enter is pressed. Nothing commands the follower before that.
  4. The follower then copies the leader, but nothing is recorded until Enter: time to bring
     both arms to `ready` (the line-up left them wherever the follower was) and place the object.
  5. Every frame at --fps: read the follower and the webcams, read the leader, step the
     follower towards it at the speed cap (arm/teach.py follow_step) and record the frame:
     observation.state and action with all six joints, plus observation.images.<cam>; the
     features lerobot-record writes, from the same LeRobot helpers (training/rehearse_sim.py
     record_features). Like lerobot-record, the action is the leader's position. A frozen
     joint is recorded as FROZEN_READING in both, never the leader's value, so the policy
     learns it never moves.

Keys, typed in the recording terminal and followed by Enter:
  Enter    ends the demo (press it the moment the claw is firmly closed on the object); before
           demo 1 and during a reset it starts the next demo.
  r        discards the demo in progress, or during its reset the one just finished, and
           records it again after a reset.
  q        stops. Every finished demo is saved, the one in its reset too; a demo still in
           progress is not (press Enter first to keep it).
  Ctrl-C   stops now: no command goes out after the current frame and the follower holds where
           it is, torque on. Saved demos are kept; the demo in progress, or one still in its
           reset, is discarded. A second Ctrl-C interrupts at once, even a save.

A demo is saved when its reset ends (--reset-time seconds, in which teleop goes on but nothing
is recorded), so that r can still discard it; there is a reset after the last demo too. With
--reset-time 0 each demo is saved as soon as it ends. When recording stops for any reason,
the follower keeps its torque on and holds its pose.

--resume adds demos to an existing dataset, after checking it was recorded with the same
cameras, sizes and fps. Without --resume an existing dataset is never touched. A new dataset
that ends up with no demos is removed again, so the same --name can simply be run again.

Speed: robot.yaml arm.move_speed_deg_s is the ceiling (--speed may lower it, never raise
it). The follower's arm joints move at most min(max_relative_target_deg, speed / fps) degrees
per frame towards the leader; the claw goes straight to the leader's, capped per command by
RealArm's max_relative_target (robot.yaml max_relative_target_deg). The printed command
applies the same step as LeRobot's max_relative_target, which caps the claw too (0-100
scale). The dataset stores the LEADER's positions, so move the leader slowly enough for the
follower to keep up.

Each demo starts at `ready` and ends the moment the claw is firmly closed on the object. The
scripted lift does the rest, so don't record it.

Cameras: LeRobot's OpenCV camera does not resize. If a webcam can't deliver exactly 320x180
at the chosen fps it fails when it opens (before the follower is touched). Check this on the
real wrist webcam in the T1 rehearsal; if it fails, record at a native size (for example
--width 640 --height 360) and resize to 320x180 in the convert step:
python -m training.convert --src data/datasets/NAME --dst data/datasets/NAME_320.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import select
import shlex
import shutil
import signal
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from arm import align as align_mod
from arm.pick import ConfigError, RealClock, close_cameras, load_arm_config
from arm.poses import JOINTS
from arm.real import FROZEN_READING
from arm.teach import _flush_stdin, follow_step

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "robot.yaml"
DATASETS = REPO_ROOT / "data" / "datasets"
TASK = "Pick up the object from the floor"
WIDTH, HEIGHT = 320, 180    # what ACT trains on (plan 10.2)
FPS = 30                    # must equal arm/pick.py RATE_HZ, the rate the policy runs at
EXE_NAME = "lerobot-record"
CONFIG_KEYS = ("follower_port", "follower_id", "leader_port", "leader_id")
SLOW_FRACTION = 0.9         # a demo recorded below this fraction of --fps gets a warning

INSTRUCTIONS = """\
Recording {episodes} demos into {root}{resume}
  - First a live table lines the arms up: put the LEADER where the follower is. The follower
    holds still; press Enter once every joint shows a tick.
  - The follower then copies the leader at up to {speed:g} deg/s ({step:.3f} deg per frame at
    {fps} fps): move the leader slowly, it lags if you go faster. Nothing is recorded yet:
    bring it to `ready`, place the object, and press Enter to start demo 1.
  - Close the claw firmly on the object, then press Enter at once to end the demo ({episode:g} s max).
  - Then a reset ({reset:g} s; the arm still follows, nothing is recorded): move the object to a
    new spot/angle inside the parking window and bring the leader back to `ready`. Enter starts
    the next demo early.
  - r + Enter: discard the demo in progress, or the one just finished (during its reset), and
    record it again. q + Enter: save the finished demos and stop.
  - Ctrl-C: stop now; the follower holds where it is and the demo not yet saved is dropped.{frozen}
  - At the end the follower keeps holding its pose (torque on): rest it or hold it before you
    switch it off.
  - Don't move any camera from now on; if one moves, the demos must be re-recorded.
"""


class RecordError(RuntimeError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


# --- robot.yaml and the lerobot-record command -----------------------------------------

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


def parse_command(cmd: list[str]):
    """The command parsed by lerobot-record's own parser: its RecordConfig. Nothing is opened."""
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    return draccus.parse(RecordConfig, args=cmd[1:])


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


# --- the dataset, exactly as lerobot-record opens it ------------------------------------

def dataset_features(rec) -> tuple[str, dict]:
    """(robot_type, features) as lerobot-record computes them for this config, with the same
    LeRobot helpers (training/rehearse_sim.py record_features): all six joints, wrist_roll
    included, and observation.images.<cam> for each camera. The follower is built from the
    config only to read its features, never connected, with a throwaway calibration folder."""
    from training.rehearse_sim import record_features

    with tempfile.TemporaryDirectory(prefix="record-features-") as calibration:
        robot, features = record_features(copy.deepcopy(rec), Path(calibration))
    return robot.name, features


def open_dataset(rec, robot_type: str, features: dict, resume: bool):
    """A new LeRobotDataset at rec.dataset.root, or with resume the existing one, opened with the
    arguments lerobot-record passes. A resumed dataset must have this recording's robot type,
    fps and features (same cameras and sizes); otherwise ValueError, and it is left as it was."""
    from lerobot.datasets import LeRobotDataset

    d = rec.dataset
    n_cams = len(rec.robot.cameras)
    if resume:
        from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility

        dataset = LeRobotDataset.resume(
            d.repo_id, root=d.root, batch_encoding_size=d.video_encoding_batch_size,
            rgb_encoder=d.rgb_encoder, depth_encoder=d.depth_encoder, encoder_threads=d.encoder_threads,
            streaming_encoding=d.streaming_encoding, encoder_queue_maxsize=d.encoder_queue_maxsize,
            image_writer_processes=d.num_image_writer_processes if n_cams else 0,
            image_writer_threads=d.num_image_writer_threads_per_camera * n_cams)
        try:
            sanity_check_dataset_robot_compatibility(dataset, SimpleNamespace(robot_type=robot_type), d.fps,
                                                     features)
        except ValueError:
            dataset.finalize()
            raise
        return dataset
    d.stamp_repo_id()
    return LeRobotDataset.create(
        d.repo_id, d.fps, root=d.root, robot_type=robot_type, features=features, use_videos=d.video,
        image_writer_processes=d.num_image_writer_processes,
        image_writer_threads=d.num_image_writer_threads_per_camera * n_cams,
        batch_encoding_size=d.video_encoding_batch_size, rgb_encoder=d.rgb_encoder,
        depth_encoder=d.depth_encoder, encoder_threads=d.encoder_threads,
        streaming_encoding=d.streaming_encoding, encoder_queue_maxsize=d.encoder_queue_maxsize)


def open_cameras(rec) -> dict:
    """{name: webcam} exactly as the command describes them (index, width, height, fps), each
    through LeRobot's OpenCV camera, the way arm/policy.py opens them for the policy
    (OpenCVFrames). If one fails to open, the ones already open are closed."""
    from arm import policy as policy_mod

    opened = {}
    try:
        for name, cam in rec.robot.cameras.items():
            opened[name] = policy_mod.OpenCVFrames(cam.index_or_path, cam.width, cam.height, cam.fps)
    except BaseException:
        close_cameras(opened)
        raise
    return opened


# --- keys and Ctrl-C ---------------------------------------------------------------

def read_key(timeout_s: float) -> str | None:
    """The line typed in the terminal, stripped and in lower case ("" for a bare Enter), if one
    is ready within timeout_s; otherwise None. Raises RecordError on end of input, or if
    stdin can't be watched (not a terminal or pipe)."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (AttributeError, OSError, ValueError) as e:      # io.UnsupportedOperation is an OSError
        raise RecordError(f"can't read keys from this input ({e}); run it in a terminal")
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        raise RecordError("input closed")
    return line.strip().lower()


class Stop:
    """Ctrl-C while recording. The first press only sets the flag, which the loop checks before
    each command, so it stops between two commands, never half-way through one. A second
    press raises KeyboardInterrupt at once (a stuck loop, or to abandon a save). Prints
    nothing itself: printing from a signal handler can break the terminal's output."""

    def __init__(self):
        self.presses = 0

    def is_set(self) -> bool:
        return self.presses > 0

    def press(self, *_) -> None:
        self.presses += 1
        if self.presses > 1:
            raise KeyboardInterrupt


def catch_ctrl_c(stop: Stop):
    """Send Ctrl-C to stop.press() until the returned function is called. Only possible in the
    main thread; anywhere else Ctrl-C stays a KeyboardInterrupt (handled the same way)."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    old = signal.signal(signal.SIGINT, stop.press)
    return lambda: signal.signal(signal.SIGINT, old)


# --- the recording loop ------------------------------------------------------------

class Recorder:
    """Teleop with the speed cap, recording demos into a LeRobotDataset (module docstring).

    leader.read() and follower.read() -> {joint: value}; follower.command({joint: value});
    cameras {name: object with read() -> H x W x 3 RGB uint8}; dataset a LeRobotDataset
    (add_frame, save_episode, clear_episode_buffer, has_pending_frames, num_episodes,
    features). keys(timeout_s) -> the line typed or None (default: the terminal). stop:
    a Stop; once set, no further command goes out and run() raises KeyboardInterrupt.
    Timing runs on follower.clock if it has one (the sim does), else the wall clock.

    The caller must line the arms up before run(): run() commands the follower from its
    first frame. On any exception the frames of an unsaved demo stay in the dataset's
    buffer; the caller discards them."""

    def __init__(self, leader, follower, cameras: dict, dataset, *, episodes: int, fps: int, max_step: float,
                 episode_time_s: float, reset_time_s: float, task: str = TASK, frozen=(), keys=None,
                 stop: Stop | None = None, log=None):
        if not (isinstance(max_step, (int, float)) and 0 < max_step < math.inf):
            raise ValueError(f"max_step must be a positive number of degrees per frame, got {max_step!r}")
        bad = [j for j in frozen if j not in JOINTS or j == "gripper"]
        if bad:
            raise ValueError(f"frozen joints must be arm joints (not the gripper), got {bad}")
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        self._build = lambda values, prefix: build_dataset_frame(dataset.features, values, prefix=prefix)
        self._obs, self._action = OBS_STR, ACTION
        self.leader, self.follower, self.cameras, self.dataset = leader, follower, cameras, dataset
        self.episodes, self.fps, self.max_step, self.task = episodes, fps, float(max_step), task
        self.demo_ticks = max(1, round(episode_time_s * fps))
        self.reset_ticks = max(0, round(reset_time_s * fps))
        self.episode_time_s, self.reset_time_s = episode_time_s, reset_time_s
        self.frozen = tuple(frozen)
        self.keys = keys or read_key
        self.stop = stop
        self.log = log or say
        self.clock = getattr(follower, "clock", None) or RealClock()
        self.cmd = None
        self.frames = 0             # frames of the current demo in the dataset's buffer
        self.saved = 0              # demos saved by this run

    # --- one frame -------------------------------------------------------------------

    def _check_stop(self) -> None:
        if self.stop is not None and self.stop.is_set():
            raise KeyboardInterrupt

    def _joints(self, reading: dict) -> dict:
        """All six joints as floats, a frozen joint at FROZEN_READING whatever was read."""
        return {j: FROZEN_READING if j in self.frozen else float(reading[j]) for j in JOINTS}

    def _frame(self, recording: bool) -> None:
        # Observe first, then act, as lerobot-record does (and pick.py runs the policy).
        state = self._joints(self.follower.read())
        images = {name: cam.read() for name, cam in self.cameras.items()} if recording else {}
        target = self._joints(self.leader.read())
        follow_step(self.cmd, target, self.max_step)
        self._check_stop()                                   # nothing goes out once Ctrl-C was pressed
        self.follower.command({j: v for j, v in self.cmd.items() if j not in self.frozen})
        if recording:
            obs = {**{f"{j}.pos": state[j] for j in JOINTS}, **images}
            action = {f"{j}.pos": target[j] for j in JOINTS}
            self.dataset.add_frame({**self._build(obs, self._obs), **self._build(action, self._action),
                                    "task": self.task})
            self.frames += 1

    def _phase(self, recording: bool, ticks: int | None) -> str:
        """Follow the leader for up to `ticks` frames at fps (None: no limit), recording them if
        `recording`. Returns "time" when they are done, or the key that ended it early:
        "enter", "r" or "q"."""
        dt = 1.0 / self.fps
        next_tick = self.clock.now()
        for _ in (itertools.count() if ticks is None else range(ticks)):
            self._check_stop()
            key = self.keys(0.0)
            if key is not None:
                key = key.strip().lower()
                if key == "":
                    return "enter"
                if key in ("r", "q"):
                    return key
                self.log(f"  '{key}' does nothing: press Enter, or type r or q and then Enter")
            self._frame(recording)
            next_tick += dt
            delay = next_tick - self.clock.now()
            if delay > 0:
                self.clock.sleep(delay)
            else:                       # running late: start the next frame now
                next_tick = self.clock.now()
        return "time"

    # --- demos ---------------------------------------------------------------------------

    def _discard(self, why: str) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self.frames = 0
        self.log(why)

    def _demo(self, number: int) -> str:
        self.log(f"\ndemo {number}/{self.episodes}: RECORDING ({self.episode_time_s:g} s at most). "
                 f"Enter: end it   r: record it again   q: stop (not saved)")
        self.frames = 0
        t0 = self.clock.now()
        end = self._phase(True, self.demo_ticks)
        seconds = self.clock.now() - t0
        if end in ("enter", "time") and self.frames:
            rate = self.frames / seconds if seconds > 0 else float(self.fps)
            how = "ended by Enter" if end == "enter" else "time up"
            self.log(f"demo {number}: {self.frames} frames, {seconds:.1f} s ({how})")
            if rate < SLOW_FRACTION * self.fps:
                self.log(f"  warning: recorded at {rate:.1f} fps, not {self.fps}: the frames are stored as "
                         f"{self.fps} fps, so this demo plays back too fast. Press r in the reset to drop it.")
        return end

    def _reset(self, number: int) -> str:
        """The reset after demo `number`: teleop, nothing recorded. The demo, if it has frames,
        waits in the buffer; r discards it and starts the reset again. Returns how the reset
        ended: "time", "enter" or "q"."""
        while True:
            pending = self.frames > 0
            if pending:
                keys = (f"Enter: save demo {number}, " + ("finish" if self.saved + 1 >= self.episodes
                                                          else "start the next one")
                        + f"   r: discard demo {number}   q: save it and stop")
            else:
                keys = "Enter: start the demo now   r: restart the reset   q: stop"
            self.log(f"reset ({self.reset_time_s:g} s, not recording): move the object, bring the leader "
                     f"back to `ready`. {keys}")
            end = self._phase(False, self.reset_ticks)
            if end != "r":
                return end
            if pending:
                self._discard(f"r: demo {number} discarded; it will be recorded again after this reset")

    def _get_ready(self) -> bool:
        """Before demo 1: teleop, nothing recorded, for as long as it takes. The line-up left the
        arms wherever the follower was; the person now brings them to `ready`. True on Enter,
        False on q."""
        self.log("\nThe follower now copies the leader; nothing is recorded yet. Bring the leader to "
                 "`ready` and place the object. Enter: start demo 1   q: stop")
        while True:
            end = self._phase(False, None)
            if end != "r":                  # nothing to discard yet
                return end == "enter"

    def run(self) -> int:
        """Record until `episodes` demos are saved or q is pressed; returns how many were saved.
        Raises KeyboardInterrupt if stopped (Ctrl-C) and any other error as it comes."""
        self.cmd = self._joints(self.follower.read())          # follow from where the follower is
        if not self._get_ready():
            self.log("q: stopping")
            return self.saved
        while self.saved < self.episodes:
            number = self.saved + 1
            end = self._demo(number)
            if end == "q":
                self._discard(f"q: stopping; demo {number} was still in progress and is not saved")
                return self.saved
            if end == "r":
                self._discard(f"r: demo {number} discarded; it will be recorded again after a reset")
            elif not self.frames:
                self.log(f"demo {number} has no frames; it will be recorded again after a reset")
            after = self._reset(number)
            if self.frames > 0:
                self._check_stop()                              # Ctrl-C in the reset: not saved
                # One camera after another: parallel encoding crashed once with two cameras
                # (training/rehearse_sim.py). The files are the same either way.
                self.dataset.save_episode(parallel_encoding=False)
                self.saved += 1
                self.frames = 0
                self.log(f"demo {number} saved ({self.dataset.num_episodes} in the dataset)")
            if after == "q":
                self.log("q: stopping")
                return self.saved
        return self.saved


# --- --run ---------------------------------------------------------------------------

def _close(thing, notes: list) -> None:
    try:
        thing.close()
    except BaseException as e:  # noqa: BLE001 - every other device must still close
        notes.append(f"closing {type(thing).__name__} failed: {type(e).__name__}: {e}")


def record_live(rec, arm_cfg: dict, max_step: float, resume: bool, log=None) -> int:
    """Run the recording the command `rec` describes, natively (module docstring). Call it only
    after the person confirmed the area is clear. Returns the exit code: 0 when every demo
    was recorded or q was pressed, 1 when stopped by Ctrl-C or an error."""
    from arm import real as real_mod

    log = log or say
    root = Path(rec.dataset.root)
    robot_type, features = dataset_features(rec)
    try:
        dataset = open_dataset(rec, robot_type, features, resume)
    except ValueError as e:
        raise SystemExit(f"can't add to {root}: it was recorded differently ({e}). Nothing was started; "
                         f"use a new --name") from None
    before = dataset.num_episodes
    frozen = tuple(arm_cfg["frozen_joints"])
    leader = follower = None
    cameras = {}
    stop = Stop()
    restore = None
    problem = None
    commanded = False
    notes = []
    discarded = False
    try:
        # Read-only devices first: if one fails, the follower is never touched.
        leader = real_mod.RealLeader(port=arm_cfg["leader_port"], robot_id=arm_cfg["leader_id"])
        cameras = open_cameras(rec)
        follower = real_mod.RealArm(port=arm_cfg["follower_port"], robot_id=arm_cfg["follower_id"],
                                    max_relative_target=arm_cfg["max_relative_target_deg"], frozen=frozen)
        _flush_stdin()
        align_mod.live_align(leader, follower)     # the follower holds still until the arms match
        _flush_stdin()                             # an extra Enter there must not end demo 1 at once
        restore = catch_ctrl_c(stop)
        commanded = True
        Recorder(leader, follower, cameras, dataset, episodes=rec.dataset.num_episodes, fps=rec.dataset.fps,
                 max_step=max_step, episode_time_s=rec.dataset.episode_time_s,
                 reset_time_s=rec.dataset.reset_time_s, task=rec.dataset.single_task, frozen=frozen,
                 keys=read_key, stop=stop, log=log).run()
    except KeyboardInterrupt:
        problem = "Ctrl-C"
    except align_mod.AlignError as e:
        problem = str(e)
    except Exception as e:  # noqa: BLE001 - a clear message, and everything still gets closed
        problem = f"{type(e).__name__}: {e}"
    finally:
        try:
            if problem and commanded:
                try:
                    log(f"\n{problem}: no more commands; the follower holds where it is. Closing, and keeping "
                        f"the saved demos (Ctrl-C again abandons that) ...")
                except BaseException:  # noqa: BLE001 - a second Ctrl-C: still close everything
                    pass
            for cam in cameras.values():
                _close(cam, notes)
            for device in (follower, leader):       # RealArm.close leaves the torque on: it holds
                if device is not None:
                    _close(device, notes)
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                    discarded = True
            except BaseException as e:  # noqa: BLE001
                notes.append(f"discarding the unsaved demo failed: {type(e).__name__}: {e}")
            try:
                dataset.finalize()
                images = root / "images"            # as lerobot-record's VideoEncodingManager: drop the
                if images.is_dir() and not any(images.rglob("*.png")) and not any(images.rglob("*.tiff")):
                    shutil.rmtree(images)           # temporary frame folders once they are empty
            except BaseException as e:  # noqa: BLE001
                notes.append(f"finalizing the dataset failed: {type(e).__name__}: {e}")
        finally:
            if restore is not None:
                restore()
    saved = dataset.num_episodes - before
    removed = not resume and dataset.num_episodes == 0 and root.exists()
    if removed:                                 # nothing recorded: let the same --name run again
        shutil.rmtree(root)
    for note in notes:
        print(f"warning: {note}", file=sys.stderr, flush=True)
    if problem:
        print(f"stopped: {problem}", file=sys.stderr, flush=True)
    log(f"\n{saved} demo(s) saved this run" + (" (an unsaved demo was discarded)" if discarded else "")
        + (f"; nothing recorded, so {root} was removed again" if removed
           else f"; {root} has {dataset.num_episodes}"))
    if follower is not None:
        log("The follower is holding its pose with the torque on: rest it or hold it before you switch it off.")
    return 1 if problem else 0


def _count(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value
    return parse


def frozen_warning(frozen: list[str]) -> str:
    return (f"warning: robot.yaml arm.frozen_joints is {frozen}, but lerobot-record builds the SO-101 "
            f"follower itself and would command {' and '.join(frozen)} anyway; a frozen servo is faulty (servo 5, "
            f"wrist_roll, browns out the whole bus when commanded). Don't run this command: record with --run, "
            f"which never calls a frozen servo.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.record",
                                description="Prints the lerobot-record command; --run records live, natively.")
    p.add_argument("--name", required=True, help="dataset name, e.g. rehearsal or grasp_v1")
    p.add_argument("--episodes", type=_count(1), default=50, help="demos to record (this run)")
    p.add_argument("--scene", action="store_true", help="also record the deck webcam as `scene` (decision D4)")
    p.add_argument("--width", type=_count(1), default=WIDTH)
    p.add_argument("--height", type=_count(1), default=HEIGHT)
    p.add_argument("--fps", type=_count(1), default=FPS)
    p.add_argument("--episode-time", type=_count(1), default=20, help="max seconds per demo")
    p.add_argument("--reset-time", type=_count(0), default=10,
                   help="seconds between demos to move the object (teleop on, nothing recorded)")
    p.add_argument("--speed", type=float, default=None,
                   help="follower speed cap in deg/s (default and maximum: robot.yaml arm.move_speed_deg_s)")
    p.add_argument("--resume", action="store_true", help="add demos to an existing dataset")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true",
                      help="record live: asks the person to confirm the area is clear, lines the arms up, "
                           "then records (never calls a frozen servo)")
    mode.add_argument("--print-only", "--dry-run", dest="print_only", action="store_true",
                      help="print the lerobot-record command, don't run anything (the default)")
    args = p.parse_args(argv)

    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name may only use letters, digits, _ and -")
    root = DATASETS / args.name
    cfg = load_config(args.config)
    speed, step = step_limit(cfg, args.speed, args.fps)
    cmd = build_command(cfg, args.name, args.episodes, ["wrist", "scene"] if args.scene else ["wrist"],
                        args.width, args.height, args.fps, args.episode_time, args.reset_time, args.resume,
                        find_lerobot_record() or EXE_NAME, speed)
    try:
        arm_cfg = load_arm_config(args.config, require=CONFIG_KEYS)
    except ConfigError as e:
        raise SystemExit(str(e)) from None
    frozen = arm_cfg["frozen_joints"]

    if not args.run:
        print(shlex.join(cmd))
        if frozen:
            print(frozen_warning(frozen), file=sys.stderr)
        print("(printed only; add --run to record live)", file=sys.stderr)
        return 0

    if args.resume and not (root / "meta" / "info.json").exists():
        raise SystemExit(f"nothing to resume at {root} (no meta/info.json); check --name")
    if root.exists() and not args.resume:
        raise SystemExit(f"{root} already exists; pick a new --name or pass --resume")
    frozen_note = "".join(f"\n  - {j} is frozen: it never moves and is recorded as {FROZEN_READING:g}."
                          for j in frozen)
    print(INSTRUCTIONS.format(episodes=args.episodes, root=root, resume=" (adding to it)" if args.resume else "",
                              episode=args.episode_time, reset=args.reset_time, speed=speed, step=step,
                              fps=args.fps, frozen=frozen_note))
    confirm_area_clear()
    DATASETS.mkdir(parents=True, exist_ok=True)
    return record_live(parse_command(cmd), arm_cfg, step, args.resume)


if __name__ == "__main__":
    sys.exit(main())
