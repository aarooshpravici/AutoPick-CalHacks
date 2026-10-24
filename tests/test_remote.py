"""training/remote.sh with stub nvidia-smi, lerobot-train, pip and python.
FP_TRAIN_HOST=local runs the PC side on this machine; the fake-ssh tests go through a stand-in
ssh that splits its command line the way a real ssh server's shell does. No GPU, no network,
no training, nothing installed. Checkpoints always land in tmp_path, never in the repo's data/."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "training" / "remote.sh"

FAKE_TRAIN = """#!/usr/bin/env bash
# Stand-in for lerobot-train: records its arguments, writes checkpoints.
nap() { trap 'kill $! 2>/dev/null; exit 143' TERM; sleep "$1" & wait $!; }   # `stop` ends the sleep too
out=""; steps=0; save=0
for a in "$@"; do
  case "$a" in --output_dir=*) out="${a#*=}";; --steps=*) steps="${a#*=}";; --save_freq=*) save="${a#*=}";; esac
done
printf '%s\\n' "$@" > "$out.args"
if [ -n "${FAKE_TRAIN_FAIL:-}" ]; then echo "ValueError: $FAKE_TRAIN_FAIL"; exit 2; fi
if [ -n "${FAKE_TRAIN_NO_CKPT:-}" ]; then mkdir -p "$out"; nap 30; exit 0; fi
for ((s=save; s<=steps; s+=save)); do
  d="$out/checkpoints/$(printf %06d $s)"
  mkdir -p "$d/pretrained_model" "$d/training_state"
  echo '{}' > "$d/pretrained_model/config.json"; echo x > "$d/training_state/optimizer.bin"
done
ln -sfn "$(printf %06d $steps)" "$out/checkpoints/last"
echo "step:$steps"
nap "${FAKE_TRAIN_SLEEP:-0}"
"""

# What sshd does with `ssh host ARGS...`: join ARGS with spaces, hand the string to the login shell.
FAKE_SSH = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host="$1"; shift
echo "$host" >> "$FAKE_SSH_LOG"
cd "$HOME" && exec bash -c "$*"
"""

FAKE_TORCH = """import os
__version__ = "2.11.0+cu128"
class cuda:
    @staticmethod
    def is_available():
        return os.environ.get("FAKE_CUDA") == "1"
    @staticmethod
    def get_device_name(i):
        return "Fake RTX 5070"
"""

REQS = "lerobot[feetech,dataset]==0.6.2\ntorch==2.11.0\ntorchvision==0.26.0\npyyaml==6.0.3\n"


def write_exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    venv_bin = home / "floor-picker-train" / "venv" / "bin"
    write_exe(venv_bin / "lerobot-train", FAKE_TRAIN)
    # Setup never builds a venv in tests: the folder exists, so python3.12 must not be called.
    write_exe(venv_bin / "pip", '#!/usr/bin/env bash\nprintf "%s\\n" "$@" --- >> "$PIP_LOG"\n')
    write_exe(venv_bin / "python", f'#!/usr/bin/env bash\nPYTHONPATH="$FAKE_PYMODS" exec {sys.executable} -s "$@"\n')
    pymods = tmp_path / "pymods"
    pymods.mkdir()
    (pymods / "torch.py").write_text(FAKE_TORCH)
    (pymods / "lerobot.py").write_text("")
    stubs = tmp_path / "stubs"
    write_exe(stubs / "nvidia-smi", '#!/usr/bin/env bash\ncat "$GPU_APPS_FILE" 2>/dev/null || true\n')
    write_exe(stubs / "python3.12", '#!/usr/bin/env bash\necho "python3.12 must not run in tests" >&2; exit 99\n')
    write_exe(stubs / "fake-ssh", FAKE_SSH)
    # Guard: if the script ever ignored FP_TRAIN_SSH, it would find this instead of the real ssh.
    write_exe(stubs / "ssh", '#!/usr/bin/env bash\necho "real ssh must not run in tests" >&2; exit 97\n')
    e = dict(os.environ, HOME=str(home), FP_TRAIN_HOST="local", PATH=f"{stubs}:{os.environ['PATH']}",
             GPU_APPS_FILE=str(tmp_path / "gpu_apps"), FP_CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
             FP_START_WAIT="1", PIP_LOG=str(tmp_path / "pip.log"), FAKE_PYMODS=str(pymods), FAKE_CUDA="1",
             FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    return e, home, tmp_path


def over_fake_ssh(e, tmp):
    return dict(e, FP_TRAIN_HOST="trainpc", FP_TRAIN_SSH=str(tmp / "stubs" / "fake-ssh"))


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    return d


def sh(env, *args, check=True, script=SCRIPT):
    r = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=60)
    if check:
        assert r.returncode == 0, r.stderr + r.stdout
    return r


def script_with_reqs(tmp, reqs):
    """A copy of remote.sh in a scratch repo whose requirements-lerobot.txt is `reqs`."""
    repo = tmp / "repo"
    (repo / "training").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "training" / "remote.sh")
    (repo / "requirements-lerobot.txt").write_text(reqs)
    return repo / "training" / "remote.sh"


def wait_for(path, timeout=10):
    t0 = time.time()
    while not path.exists():
        assert time.time() - t0 < timeout, f"{path} never appeared"
        time.sleep(0.05)


def trainer_args(root, name):
    wait_for(root / "outputs" / f"{name}.args")
    return (root / "outputs" / f"{name}.args").read_text().splitlines()


def check_tfs(args):
    tfs = [a.split("=", 1)[1] for a in args if a.startswith("--dataset.image_transforms.tfs=")]
    assert len(tfs) == 1, args
    cfg = json.loads(tfs[0])                                           # valid JSON, not shell-mangled
    assert cfg["brightness"]["kwargs"] == {"brightness": [0.8, 1.2]}
    assert cfg["contrast"]["kwargs"] == {"contrast": [0.8, 1.2]}


def test_needs_host(dataset):
    r = subprocess.run(["bash", str(SCRIPT), "status", "x"], env={k: v for k, v in os.environ.items() if k != "FP_TRAIN_HOST"},
                       capture_output=True, text=True)
    assert r.returncode != 0 and "FP_TRAIN_HOST" in r.stderr


def test_help_needs_no_host():
    r = subprocess.run(["bash", str(SCRIPT), "--help"], env={k: v for k, v in os.environ.items() if k != "FP_TRAIN_HOST"},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "FP_CHECKPOINT_DIR" in r.stdout and "set -euo" not in r.stdout


def test_refuses_busy_gpu(env, dataset):
    e, home, tmp = env
    (tmp / "gpu_apps").write_text("4242, python, 9000 MiB\n")
    r = sh(e, "run", str(dataset), "rehearsal", "--steps", "200", check=False)
    assert r.returncode != 0 and "in use" in r.stderr
    assert not (home / "floor-picker-train" / "outputs" / "rehearsal").exists()


def test_force_still_reports_gpu_users(env, dataset):
    e, home, tmp = env
    (tmp / "gpu_apps").write_text("4242, python, 9000 MiB\n")
    r = sh(e, "run", str(dataset), "mine", "--steps", "200", "--force")
    assert "4242, python" in r.stderr


def test_rehearsal_run_status_pull(env, dataset):
    e, home, tmp = env
    name = "pytest-rehearsal"
    out = sh(e, "run", str(dataset), name, "--steps", "200").stdout
    assert "200 steps" in out
    root = home / "floor-picker-train"
    args = trainer_args(root, name)
    assert "--policy.type=act" in args and "--steps=200" in args and "--save_freq=200" in args
    assert "--dataset.image_transforms.enable=true" in args
    check_tfs(args)
    assert (root / "datasets" / name / "meta" / "info.json").exists()

    wait_for(root / "outputs" / name / "checkpoints" / "last")
    assert "000200" in sh(e, "status", name).stdout

    pulled = sh(e, "pull", name).stdout
    dest = tmp / "checkpoints" / name
    assert (dest / "000200" / "pretrained_model" / "config.json").exists()
    assert not (dest / "000200" / "training_state").exists()       # optimizer state stays on the PC
    assert not (dest / "last").exists()
    assert "pretrained_model" in pulled
    assert not (REPO / "data" / "checkpoints" / name).exists()      # the repo's data/ is never touched


def test_run_over_ssh_keeps_arguments_intact(env, dataset):
    """ssh re-splits its command line on the PC; every argument must still arrive whole."""
    e, home, tmp = env
    e = over_fake_ssh(e, tmp)
    out = sh(e, "run", str(dataset), "r-ssh", "--steps", "200").stdout
    assert "200 steps" in out
    root = home / "floor-picker-train"
    args = trainer_args(root, "r-ssh")
    check_tfs(args)
    for a in ("--dataset.repo_id=local/r-ssh", f"--dataset.root={root}/datasets/r-ssh", "--policy.type=act",
              "--policy.device=cuda", f"--output_dir={root}/outputs/r-ssh", "--job_name=r-ssh", "--steps=200",
              "--save_freq=200", "--dataset.image_transforms.max_num_transforms=2"):
        assert a in args, (a, args)
    assert (root / "datasets" / "r-ssh" / "meta" / "info.json").exists()    # rsync went through the fake ssh too
    wait_for(root / "outputs" / "r-ssh" / "checkpoints" / "last")
    assert "000200" in sh(e, "status", "r-ssh").stdout
    sh(e, "pull", "r-ssh")
    assert (tmp / "checkpoints" / "r-ssh" / "000200" / "pretrained_model" / "config.json").exists()
    assert set((tmp / "ssh.log").read_text().split()) == {"trainpc"}


def test_lerobot_train_accepts_the_flags(env, dataset):
    """remote.sh's lerobot-train flags, read by lerobot-train's own config parser (LeRobot 0.6.2),
    alone and with the Mac changes training/rehearse_sim.py adds."""
    pytest.importorskip("lerobot.scripts.lerobot_train")
    import draccus
    from lerobot.scripts.lerobot_train import TrainPipelineConfig
    from training import rehearse_sim

    e, home, tmp = env
    sh(e, "run", str(dataset), "flags", "--steps", "200")
    args = trainer_args(home / "floor-picker-train", "flags")
    cfg = draccus.parse(TrainPipelineConfig, args=args)
    assert (cfg.policy.type, cfg.policy.push_to_hub, cfg.wandb.enable) == ("act", False, False)
    assert (cfg.steps, cfg.save_freq, cfg.dataset.repo_id) == (200, 200, "local/flags")
    tfs = cfg.dataset.image_transforms
    assert tfs.enable and tfs.max_num_transforms == 2 and sorted(tfs.tfs) == ["brightness", "contrast"]
    assert tfs.tfs["brightness"].type == "ColorJitter" and tfs.tfs["contrast"].kwargs == {"contrast": [0.8, 1.2]}
    mac = draccus.parse(TrainPipelineConfig, args=args + rehearse_sim.mac_overrides(200))
    assert mac.policy.pretrained_backbone_weights is None
    assert (mac.batch_size, mac.num_workers) == (2, 0)


def test_setup_over_ssh_quotes_arguments(env):
    e, home, tmp = env
    index = "https://wheels.example.invalid/cu 128/it's {x}"
    e = dict(over_fake_ssh(e, tmp), FP_TORCH_INDEX=index)
    out = sh(e, "setup", script=script_with_reqs(tmp, REQS)).stdout
    assert "cuda available: True" in out and "Fake RTX 5070" in out
    calls = [c.strip("\n").split("\n") for c in (tmp / "pip.log").read_text().split("---\n") if c.strip()]
    assert ["install", "-q", "torch==2.11.0", "torchvision==0.26.0", "--index-url", index] in calls
    assert ["install", "-q", "-r", "requirements-lerobot.txt"] in calls
    assert (home / "floor-picker-train" / "requirements-lerobot.txt").read_text() == REQS


def test_setup_fails_without_cuda(env):
    e, _, tmp = env
    r = sh(dict(e, FAKE_CUDA="0"), "setup", script=script_with_reqs(tmp, REQS), check=False)
    assert r.returncode != 0 and "CUDA is not available" in r.stderr


def test_setup_needs_torch_pins(env):
    e, home, tmp = env
    r = sh(e, "setup", script=script_with_reqs(tmp, "lerobot==0.6.2\n"), check=False)
    assert r.returncode != 0 and "torch==" in r.stderr
    assert not (tmp / "pip.log").exists() and not (home / "floor-picker-train" / "requirements-lerobot.txt").exists()


def test_run_reports_startup_crash(env, dataset):
    e, _, _ = env
    r = sh(dict(e, FAKE_TRAIN_FAIL="bad image_transforms.tfs"), "run", str(dataset), "crash", "--steps", "200", check=False)
    assert r.returncode != 0
    assert "exited with code 2" in r.stderr and "ValueError: bad image_transforms.tfs" in r.stderr
    assert "started" not in r.stdout
    assert "ValueError" in sh(e, "status", "crash").stdout            # status shows how it ended


def test_run_needs_setup_first(env, dataset):
    e, home, _ = env
    (home / "floor-picker-train" / "venv" / "bin" / "lerobot-train").unlink()
    r = sh(e, "run", str(dataset), "nosetup", "--steps", "200", check=False)
    assert r.returncode != 0 and "remote.sh setup" in r.stderr
    assert not (home / "floor-picker-train" / "datasets" / "nosetup").exists()


def test_pull_before_first_checkpoint(env, dataset):
    e, home, tmp = env
    e = dict(e, FAKE_TRAIN_NO_CKPT="1")
    sh(e, "run", str(dataset), "early", "--steps", "200")
    try:
        r = sh(e, "pull", "early")
        assert "no checkpoints yet" in r.stdout
        assert not (tmp / "checkpoints" / "early").exists()
    finally:
        sh(e, "stop", "early")


def test_same_name_twice_refused(env, dataset):
    e, _, _ = env
    sh(e, "run", str(dataset), "r1", "--steps", "200")
    r = sh(e, "run", str(dataset), "r1", "--steps", "200", check=False)
    assert r.returncode != 0


def test_stop(env, dataset):
    e, home, _ = env
    e = dict(e, FAKE_TRAIN_SLEEP="30")
    assert "started" in sh(e, "run", str(dataset), "long", "--steps", "200").stdout
    assert "running" in sh(e, "status", "long").stdout.splitlines()[0]
    assert "stopped" in sh(e, "stop", "long").stdout
    time.sleep(0.2)
    assert sh(e, "status", "long").stdout.startswith("not running")


def test_stop_leaves_reused_pid_alone(env):
    """The pid file points at a live process that is not this run's lerobot-train."""
    e, home, _ = env
    pids = home / "floor-picker-train" / "pids"
    pids.mkdir(parents=True)
    other = subprocess.Popen(["sleep", "30"])
    try:
        (pids / "old.pid").write_text(f"{other.pid}\n")
        assert sh(e, "status", "old").stdout.startswith("not running")
        out = sh(e, "stop", "old").stdout
        assert "not running" in out and "left alone" in out
        time.sleep(0.2)
        assert other.poll() is None                                     # still alive
    finally:
        other.kill()
        other.wait()


def test_rejects_bad_name(env, dataset):
    e, _, _ = env
    assert sh(e, "run", str(dataset), "../evil", check=False).returncode != 0


@pytest.mark.parametrize("steps", [["--steps", "200; touch pwned"], ["--steps", "abc"], ["--steps", "0"], ["--steps"]])
def test_rejects_bad_steps(env, dataset, steps):
    e, home, _ = env
    r = sh(e, "run", str(dataset), "s1", *steps, check=False)
    assert r.returncode != 0 and "--steps" in r.stderr
    assert not (home / "floor-picker-train" / "datasets").exists()


def test_rejects_unsafe_root(env, dataset):
    e, _, _ = env
    r = sh(dict(e, FP_TRAIN_ROOT="train dir; rm -rf x"), "status", "x", check=False)
    assert r.returncode != 0 and "FP_TRAIN_ROOT" in r.stderr


def test_rejects_non_dataset(env, tmp_path):
    e, _, _ = env
    r = sh(e, "run", str(tmp_path), "x", check=False)
    assert r.returncode != 0 and "LeRobot dataset" in r.stderr
