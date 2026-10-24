"""training/remote.sh in FP_TRAIN_HOST=local mode, with stub nvidia-smi and lerobot-train.
No GPU, no SSH, no training."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "training" / "remote.sh"

FAKE_TRAIN = """#!/usr/bin/env bash
# Stand-in for lerobot-train: records its arguments, writes checkpoints.
out=""; steps=0; save=0
for a in "$@"; do
  case "$a" in --output_dir=*) out="${a#*=}";; --steps=*) steps="${a#*=}";; --save_freq=*) save="${a#*=}";; esac
done
printf '%s\\n' "$@" > "$out.args"
for ((s=save; s<=steps; s+=save)); do
  d="$out/checkpoints/$(printf %06d $s)"
  mkdir -p "$d/pretrained_model" "$d/training_state"
  echo '{}' > "$d/pretrained_model/config.json"; echo x > "$d/training_state/optimizer.bin"
done
ln -sfn "$(printf %06d $steps)" "$out/checkpoints/last"
echo "step:$steps"
sleep "${FAKE_TRAIN_SLEEP:-0}"
"""


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    venv_bin = home / "floor-picker-train" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "lerobot-train").write_text(FAKE_TRAIN)
    (venv_bin / "lerobot-train").chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    smi = stubs / "nvidia-smi"
    smi.write_text('#!/usr/bin/env bash\ncat "$GPU_APPS_FILE" 2>/dev/null || true\n')
    smi.chmod(0o755)
    e = dict(os.environ, HOME=str(home), FP_TRAIN_HOST="local", PATH=f"{stubs}:{os.environ['PATH']}",
             GPU_APPS_FILE=str(tmp_path / "gpu_apps"))
    return e, home, tmp_path


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    return d


def sh(env, *args, check=True):
    r = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)
    if check:
        assert r.returncode == 0, r.stderr + r.stdout
    return r


def wait_for(path, timeout=10):
    t0 = time.time()
    while not path.exists():
        assert time.time() - t0 < timeout, f"{path} never appeared"
        time.sleep(0.05)


def test_needs_host(dataset):
    r = subprocess.run(["bash", str(SCRIPT), "status", "x"], env={k: v for k, v in os.environ.items() if k != "FP_TRAIN_HOST"},
                       capture_output=True, text=True)
    assert r.returncode != 0 and "FP_TRAIN_HOST" in r.stderr


def test_refuses_busy_gpu(env, dataset):
    e, home, tmp = env
    (tmp / "gpu_apps").write_text("4242, python, 9000 MiB\n")
    r = sh(e, "run", str(dataset), "rehearsal", "--steps", "200", check=False)
    assert r.returncode != 0 and "in use" in r.stderr
    assert not (home / "floor-picker-train" / "outputs" / "rehearsal").exists()


def test_rehearsal_run_status_pull(env, dataset):
    e, home, _ = env
    out = sh(e, "run", str(dataset), "rehearsal", "--steps", "200").stdout
    assert "200 steps" in out
    root = home / "floor-picker-train"
    wait_for(root / "outputs" / "rehearsal.args")
    args = (root / "outputs" / "rehearsal.args").read_text().splitlines()
    assert "--policy.type=act" in args and "--steps=200" in args and "--save_freq=200" in args
    assert "--dataset.image_transforms.enable=true" in args
    assert any(a.startswith("--dataset.image_transforms.tfs=") and "contrast" in a for a in args)
    assert (root / "datasets" / "rehearsal" / "meta" / "info.json").exists()

    wait_for(root / "outputs" / "rehearsal" / "checkpoints" / "last")
    assert "000200" in sh(e, "status", "rehearsal").stdout

    pulled = sh(e, "pull", "rehearsal").stdout
    dest = REPO / "data" / "checkpoints" / "rehearsal"
    try:
        assert (dest / "000200" / "pretrained_model" / "config.json").exists()
        assert not (dest / "000200" / "training_state").exists()       # optimizer state stays on the PC
        assert not (dest / "last").exists()
        assert "pretrained_model" in pulled
    finally:
        subprocess.run(["rm", "-rf", str(dest)])


def test_same_name_twice_refused(env, dataset):
    e, _, _ = env
    sh(e, "run", str(dataset), "r1", "--steps", "200")
    r = sh(e, "run", str(dataset), "r1", "--steps", "200", check=False)
    assert r.returncode != 0


def test_stop(env, dataset):
    e, home, _ = env
    e = dict(e, FAKE_TRAIN_SLEEP="30")
    sh(e, "run", str(dataset), "long", "--steps", "200")
    assert "running" in sh(e, "status", "long").stdout.splitlines()[0]
    assert "stopped" in sh(e, "stop", "long").stdout
    time.sleep(0.2)
    assert sh(e, "status", "long").stdout.startswith("not running")


def test_rejects_bad_name(env, dataset):
    e, _, _ = env
    assert sh(e, "run", str(dataset), "../evil", check=False).returncode != 0


def test_rejects_non_dataset(env, tmp_path):
    e, _, _ = env
    r = sh(e, "run", str(tmp_path), "x", check=False)
    assert r.returncode != 0 and "LeRobot dataset" in r.stderr
