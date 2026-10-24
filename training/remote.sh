#!/usr/bin/env bash
# Train ACT on the Ubuntu PC over SSH (task T2, plan 10.2).
#
#   training/remote.sh setup                            pinned venv on the PC (CUDA torch)
#   training/remote.sh run DATASET_DIR NAME [--steps N] [--force]
#                                                       copy the dataset, start training
#   training/remote.sh status NAME                      progress and checkpoints
#   training/remote.sh pull NAME                        copy new checkpoints to data/checkpoints/NAME/
#   training/remote.sh stop NAME                        stop the run
#
# Settings (environment):
#   FP_TRAIN_HOST   user@host of the PC (required; "local" runs everything on this machine, for tests)
#   FP_TRAIN_ROOT   folder on the PC, relative to its home (default floor-picker-train)
#   FP_TORCH_INDEX  PyTorch wheel index for the PC (default cu128, needed for the RTX 5070)
#
# The PC is shared: `run` refuses to start if any process is on the GPU, unless --force.
set -euo pipefail

HOST="${FP_TRAIN_HOST:-}"
ROOT="${FP_TRAIN_ROOT:-floor-picker-train}"
TORCH_INDEX="${FP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REQS="$REPO/requirements-lerobot.txt"

die() { echo "remote.sh: $*" >&2; exit 1; }
[ -n "$HOST" ] || die "set FP_TRAIN_HOST=user@host (the training PC)"

# Run a bash script (stdin) on the PC, in its home folder, with arguments.
on_pc() {
  if [ "$HOST" = local ]; then (cd "$HOME" && bash -s -- "$@"); else ssh -o BatchMode=yes "$HOST" bash -s -- "$@"; fi
}
pc_path() { if [ "$HOST" = local ]; then echo "$HOME/$1"; else echo "$HOST:$1"; fi; }

name_arg() {
  [ -n "${1:-}" ] || die "give the run a NAME"
  [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || die "NAME may only use letters, digits, . _ -"
  echo "$1"
}

cmd_setup() {
  [ -f "$REQS" ] || die "missing $REQS (task S3)"
  on_pc "$ROOT" <<<'mkdir -p "$1"'
  rsync -a "$REQS" "$(pc_path "$ROOT/requirements-lerobot.txt")"
  local torch tv
  torch=$(grep -E '^torch==' "$REQS" | cut -d= -f3)
  tv=$(grep -E '^torchvision==' "$REQS" | cut -d= -f3)
  on_pc "$ROOT" "$torch" "$tv" "$TORCH_INDEX" <<'EOF'
set -euo pipefail
cd "$1"
command -v python3.12 >/dev/null || { echo "python3.12 is not installed on the PC"; exit 1; }
[ -d venv ] || python3.12 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q "torch==$2" "torchvision==$3" --index-url "$4"
venv/bin/pip install -q -r requirements-lerobot.txt
venv/bin/python - <<'PY'
import torch, lerobot
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY
EOF
}

cmd_run() {
  local dataset="${1:-}"; shift || true
  local name; name=$(name_arg "${1:-}"); shift || true
  local steps=20000 force=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --steps) steps="$2"; shift 2 ;;
      --force) force=1; shift ;;
      *) die "unknown option $1" ;;
    esac
  done
  [ -f "$dataset/meta/info.json" ] || die "$dataset is not a LeRobot dataset (no meta/info.json)"
  local save=$(( steps < 5000 ? steps : 5000 ))

  if [ "$force" = 0 ]; then
    local busy
    busy=$(on_pc <<<'command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "NO_NVIDIA_SMI"')
    [ "$busy" != NO_NVIDIA_SMI ] || die "nvidia-smi not found on the PC"
    [ -z "$busy" ] || die "the GPU is in use (someone else's job?):
$busy
Not starting. Ask whoever owns it, or pass --force if it's yours."
  fi

  on_pc "$ROOT" "$name" <<<'mkdir -p "$1/datasets" "$1/outputs" "$1/logs" "$1/pids"; [ ! -e "$1/pids/$2.pid" ] && [ ! -e "$1/outputs/$2" ] || { echo "run $2 already exists on the PC; pick a new NAME" >&2; exit 1; }'
  rsync -a --delete "$dataset/" "$(pc_path "$ROOT/datasets/$name/")"

  local tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}}'
  on_pc "$ROOT" "$name" "$steps" "$save" "$tfs" <<'EOF'
set -euo pipefail
cd "$1"
nohup venv/bin/lerobot-train \
  --dataset.repo_id="local/$2" --dataset.root="$PWD/datasets/$2" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$PWD/outputs/$2" --job_name="$2" \
  --steps="$3" --save_freq="$4" --wandb.enable=false \
  --dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.tfs="$5" \
  > "logs/$2.log" 2>&1 &
echo $! > "pids/$2.pid"
echo "started $2 (pid $(cat "pids/$2.pid")): $3 steps, checkpoint every $4"
EOF
}

cmd_status() {
  local name; name=$(name_arg "${1:-}")
  on_pc "$ROOT" "$name" <<'EOF'
cd "$1" || exit 1
pid=$(cat "pids/$2.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "running (pid $pid)"; else echo "not running"; fi
echo "checkpoints: $(ls "outputs/$2/checkpoints" 2>/dev/null | grep -v last | tr '\n' ' ')"
grep -E "step:|Error|error" "logs/$2.log" 2>/dev/null | tail -3 || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
EOF
}

cmd_pull() {
  local name; name=$(name_arg "${1:-}")
  local dest="$REPO/data/checkpoints/$name"
  mkdir -p "$dest"
  # Only each checkpoint's pretrained_model folder (about 200 MB), not optimizer state.
  rsync -a --prune-empty-dirs --exclude='last' --include='*/' --include='*/pretrained_model/**' --exclude='*' \
    "$(pc_path "$ROOT/outputs/$name/checkpoints/")" "$dest/"
  ls -d "$dest"/*/pretrained_model 2>/dev/null || echo "no checkpoints yet"
}

cmd_stop() {
  local name; name=$(name_arg "${1:-}")
  on_pc "$ROOT" "$name" <<'EOF'
cd "$1" || exit 1
pid=$(cat "pids/$2.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid"; echo "stopped $2 (pid $pid)"; else echo "$2 is not running"; fi
EOF
}

cmd="${1:-}"; shift || true
case "$cmd" in
  setup) cmd_setup "$@" ;;
  run) cmd_run "$@" ;;
  status) cmd_status "$@" ;;
  pull) cmd_pull "$@" ;;
  stop) cmd_stop "$@" ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
