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
#   FP_TRAIN_HOST      user@host of the PC (required; "local" runs everything on this machine, for tests)
#   FP_TRAIN_ROOT      folder on the PC, relative to its home (default floor-picker-train)
#   FP_TORCH_INDEX     PyTorch wheel index for the PC (default cu128, needed for the RTX 5070)
#   FP_CHECKPOINT_DIR  where `pull` puts checkpoints (default data/checkpoints in this repo)
#   FP_START_WAIT      seconds `run` watches lerobot-train for a startup crash (default 15)
#   FP_TRAIN_SSH       ssh program (default ssh; tests point it at a fake, no spaces in its path)
#
# The PC is shared: `run` refuses to start if any process is on the GPU, unless --force.
set -euo pipefail

HOST="${FP_TRAIN_HOST:-}"
ROOT="${FP_TRAIN_ROOT:-floor-picker-train}"
TORCH_INDEX="${FP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SSH="${FP_TRAIN_SSH:-ssh}"
START_WAIT="${FP_START_WAIT:-15}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REQS="$REPO/requirements-lerobot.txt"
CKPT_DIR="${FP_CHECKPOINT_DIR:-$REPO/data/checkpoints}"

die() { echo "remote.sh: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1' "$0"; }

# Shell functions every PC-side script below can use (on_pc puts them in front of the script).
PC_LIB='
# ours PID NAME: PID is alive and is the lerobot-train of run NAME, started from this folder.
# Guards against a reused PID on the shared PC.
ours() {
  local args
  [[ "${1:-}" =~ ^[0-9]+$ ]] || return 1
  args=$(ps -ww -p "$1" -o args= 2>/dev/null) || return 1
  case " $args " in *lerobot-train*" --output_dir=$PWD/outputs/$2 "*) return 0 ;; esac
  return 1
}
'

# Run a bash script (stdin) on the PC, in its home folder, with arguments.
# ssh joins its arguments into one string that the PC's shell splits again, so each one is quoted.
on_pc() {
  # The script goes in through a pipe, as it does over ssh: from a seekable file, bash rewinds
  # its input when it forks a background job and runs the rest of the script twice.
  local script quoted=""
  script="$PC_LIB$(cat)"
  if [ "$HOST" = local ]; then
    printf '%s\n' "$script" | (cd "$HOME" && bash -s -- "$@")
  else
    [ $# -eq 0 ] || quoted=$(printf ' %q' "$@")
    printf '%s\n' "$script" | "$SSH" -o BatchMode=yes "$HOST" "bash -s --$quoted"
  fi
}
pc_path() { if [ "$HOST" = local ]; then echo "$HOME/$1"; else echo "$HOST:$1"; fi; }
rsync_pc() { if [ "$HOST" = local ]; then rsync "$@"; else rsync -e "$SSH -o BatchMode=yes" "$@"; fi; }

name_arg() {
  [ -n "${1:-}" ] || die "give the run a NAME"
  [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || die "NAME may only use letters, digits, . _ -"
  echo "$1"
}

cmd_setup() {
  [ -f "$REQS" ] || die "missing $REQS (task S3)"
  local torch tv
  torch=$(sed -n 's/^torch==\([^ ;#]*\).*/\1/p' "$REQS")
  tv=$(sed -n 's/^torchvision==\([^ ;#]*\).*/\1/p' "$REQS")
  [ -n "$torch" ] && [ -n "$tv" ] || die "$REQS must pin torch==VERSION and torchvision==VERSION"
  on_pc "$ROOT" <<<'mkdir -p "$1"'
  rsync_pc -a "$REQS" "$(pc_path "$ROOT/requirements-lerobot.txt")"
  on_pc "$ROOT" "$torch" "$tv" "$TORCH_INDEX" <<'EOF'
set -euo pipefail
cd "$1"
command -v python3.12 >/dev/null || { echo "python3.12 is not installed on the PC"; exit 1; }
[ -d venv ] || python3.12 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q "torch==$2" "torchvision==$3" --index-url "$4"
venv/bin/pip install -q -r requirements-lerobot.txt
venv/bin/python - <<'PY'
import sys
import torch, lerobot
cuda = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda available:", cuda,
      "|", torch.cuda.get_device_name(0) if cuda else "no GPU")
if not cuda:
    sys.exit("CUDA is not available to torch on the PC, so training would run on the CPU. "
             "Check the NVIDIA driver and FP_TORCH_INDEX, then run setup again.")
PY
EOF
}

cmd_run() {
  local dataset="${1:-}"; shift || true
  local name; name=$(name_arg "${1:-}"); shift || true
  local steps=20000 force=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --steps) [ $# -ge 2 ] || die "--steps needs a number"; steps="$2"; shift 2 ;;
      --force) force=1; shift ;;
      *) die "unknown option $1" ;;
    esac
  done
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive whole number, not '$steps'"
  [[ "$START_WAIT" =~ ^[0-9]+$ ]] || die "FP_START_WAIT must be a whole number of seconds"
  [ -f "$dataset/meta/info.json" ] || die "$dataset is not a LeRobot dataset (no meta/info.json)"
  local save=$(( steps < 5000 ? steps : 5000 ))

  on_pc "$ROOT" "$name" <<'EOF'
mkdir -p "$1/datasets" "$1/outputs" "$1/logs" "$1/pids"
[ -x "$1/venv/bin/lerobot-train" ] || { echo "lerobot-train is not installed on the PC; run: training/remote.sh setup" >&2; exit 1; }
[ ! -e "$1/pids/$2.pid" ] && [ ! -e "$1/outputs/$2" ] || { echo "run $2 already exists on the PC; pick a new NAME" >&2; exit 1; }
EOF

  local busy
  busy=$(on_pc <<<'command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "NO_NVIDIA_SMI"')
  if [ "$force" = 0 ]; then
    [ "$busy" != NO_NVIDIA_SMI ] || die "nvidia-smi not found on the PC"
    [ -z "$busy" ] || die "the GPU is in use (someone else's job?):
$busy
Not starting. Ask whoever owns it, or pass --force if it's yours."
  elif [ "$busy" = NO_NVIDIA_SMI ]; then
    echo "remote.sh: --force: nvidia-smi not found on the PC, so the GPU was not checked" >&2
  elif [ -n "$busy" ]; then
    echo "remote.sh: --force: starting even though these processes are on the GPU:
$busy" >&2
  fi

  rsync_pc -a --delete "$dataset/" "$(pc_path "$ROOT/datasets/$name/")"

  on_pc "$ROOT" "$name" "$steps" "$save" "$START_WAIT" <<'EOF'
set -euo pipefail
cd "$1"
# Image augmentation (plan 10.2): brightness and contrast jitter. Kept here, inside the PC-side
# script, so the JSON is never re-split by a shell on the way over.
tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}}'
nohup venv/bin/lerobot-train \
  --dataset.repo_id="local/$2" --dataset.root="$PWD/datasets/$2" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$PWD/outputs/$2" --job_name="$2" \
  --steps="$3" --save_freq="$4" --wandb.enable=false \
  --dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.tfs="$tfs" \
  < /dev/null > "logs/$2.log" 2>&1 &
pid=$!
echo "$pid" > "pids/$2.pid"
# A bad argument, a broken install or a bad dataset makes lerobot-train exit within seconds.
for ((i = 0; i < $5 * 5; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
if kill -0 "$pid" 2>/dev/null; then
  echo "started $2 (pid $pid): $3 steps, checkpoint every $4"
  exit 0
fi
rc=0; wait "$pid" || rc=$?
if [ "$rc" != 0 ]; then
  echo "lerobot-train for $2 exited with code $rc right after starting. Last lines of logs/$2.log on the PC:" >&2
  tail -n 20 "logs/$2.log" >&2
  exit 1
fi
echo "$2 already finished (pid $pid, exit 0): $3 steps, checkpoint every $4"
EOF
}

cmd_status() {
  local name; name=$(name_arg "${1:-}")
  on_pc "$ROOT" "$name" <<'EOF'
cd "$1" || exit 1
pid=$(cat "pids/$2.pid" 2>/dev/null || true)
if ours "$pid" "$2"; then running=1; echo "running (pid $pid)"; else running=0; echo "not running"; fi
echo "checkpoints: $(ls "outputs/$2/checkpoints" 2>/dev/null | grep -v last | tr '\n' ' ')"
if [ "$running" = 1 ]; then
  grep -E "step:|Error|error" "logs/$2.log" 2>/dev/null | tail -3 || true
else
  tail -n 5 "logs/$2.log" 2>/dev/null || true   # how it ended: finished, or the error
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
EOF
}

cmd_pull() {
  local name; name=$(name_arg "${1:-}")
  local has
  has=$(on_pc "$ROOT" "$name" <<<'[ -d "$1/outputs/$2/checkpoints" ] && echo HAS_CHECKPOINTS || echo NO_CHECKPOINTS')
  case "$has" in
    *HAS_CHECKPOINTS) ;;
    *NO_CHECKPOINTS) echo "no checkpoints yet"; return 0 ;;
    *) die "could not check $name's checkpoints on the PC" ;;
  esac
  local dest="$CKPT_DIR/$name"
  mkdir -p "$dest"
  # Only each checkpoint's pretrained_model folder (about 200 MB), not optimizer state.
  rsync_pc -a --prune-empty-dirs --exclude='last' --include='*/' --include='*/pretrained_model/**' --exclude='*' \
    "$(pc_path "$ROOT/outputs/$name/checkpoints/")" "$dest/"
  ls -d "$dest"/*/pretrained_model 2>/dev/null || echo "no checkpoints yet"
}

cmd_stop() {
  local name; name=$(name_arg "${1:-}")
  on_pc "$ROOT" "$name" <<'EOF'
cd "$1" || exit 1
pid=$(cat "pids/$2.pid" 2>/dev/null || true)
if ours "$pid" "$2"; then
  kill "$pid"; echo "stopped $2 (pid $pid)"
elif [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "$2 is not running (pid $pid is now another process; left alone)"
else
  echo "$2 is not running"
fi
EOF
}

cmd="${1:-}"; shift || true
case "$cmd" in -h|--help|help) usage; exit 0 ;; esac
[ -n "$HOST" ] || die "set FP_TRAIN_HOST=user@host (the training PC)"
[[ "$HOST" != -* ]] || die "FP_TRAIN_HOST must be user@host, not an option"
[[ "$ROOT" =~ ^[A-Za-z0-9_.][A-Za-z0-9_./-]*$ ]] || die "FP_TRAIN_ROOT may only use letters, digits, . _ - /"
case "$cmd" in
  setup) cmd_setup "$@" ;;
  run) cmd_run "$@" ;;
  status) cmd_status "$@" ;;
  pull) cmd_pull "$@" ;;
  stop) cmd_stop "$@" ;;
  *) usage; exit 1 ;;
esac
