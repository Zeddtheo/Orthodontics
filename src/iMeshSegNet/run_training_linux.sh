#!/usr/bin/env bash
# One-click launcher for iMeshSegNet training on Linux using nohup.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
RUN_NAME="$(date +%Y%m%d_%H%M%S)"
DATA_ROOT=""
SKIP_MODULE0=0
TRAIN_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_training_linux.sh [options] [-- additional-train-args...]

Options:
  --run-name NAME          Manual run identifier (default: current timestamp).
  --python PATH            Python executable to use (default: env $PYTHON or "python").
  --data-root PATH         Dataset root directory for module0 (default: datasets/segmentation_dataset).
  --skip-module0           Skip running module0 dataset preparation.
  -h, --help               Show this message and exit.

Anything after "--" is forwarded verbatim to module1 (training) CLI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --skip-module0)
      SKIP_MODULE0=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$PROJECT_ROOT/datasets/segmentation_dataset"
fi

LOG_ROOT="$PROJECT_ROOT/outputs/segmentation/logs"
CKPT_ROOT="$PROJECT_ROOT/outputs/segmentation/checkpoints"
TB_ROOT="$PROJECT_ROOT/outputs/segmentation/tensorboard"

LOG_DIR="$LOG_ROOT/$RUN_NAME"
CKPT_DIR="$CKPT_ROOT/$RUN_NAME"
TB_DIR="$TB_ROOT/$RUN_NAME"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$TB_DIR"

echo "[$(date '+%F %T')] Run name       : $RUN_NAME"
echo "[$(date '+%F %T')] Project root   : $PROJECT_ROOT"
echo "[$(date '+%F %T')] Dataset root   : $DATA_ROOT"
echo "[$(date '+%F %T')] Log directory  : $LOG_DIR"
echo "[$(date '+%F %T')] Checkpoints dir: $CKPT_DIR"
echo "[$(date '+%F %T')] TensorBoard dir: $TB_DIR"

if [[ "$SKIP_MODULE0" -eq 0 ]]; then
  DATA_LOG="$LOG_DIR/module0_dataset.log"
  echo "[$(date '+%F %T')] Running module0 dataset preparation..."
  "$PYTHON_BIN" -m src.iMeshSegNet.m0_dataset --root "$DATA_ROOT" > "$DATA_LOG" 2>&1
  echo "[$(date '+%F %T')] Module0 finished. Log: $DATA_LOG"
else
  echo "[$(date '+%F %T')] Skipping module0 dataset preparation."
fi

TRAIN_LOG="$LOG_DIR/module1_train.log"
echo "[$(date '+%F %T')] Launching module1 training in background..."

TRAIN_CMD=(
  "$PYTHON_BIN" -m src.iMeshSegNet.m1_train
  --run-name "$RUN_NAME"
  --log-dir "$LOG_DIR"
  --checkpoint-dir "$CKPT_DIR"
  --tensorboard-dir "$TB_DIR"
)
if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${TRAIN_EXTRA_ARGS[@]}")
fi

nohup "${TRAIN_CMD[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$LOG_DIR/module1_train.pid"

sleep 1
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
  echo "[$(date '+%F %T')] Training started (PID: $TRAIN_PID)."
  echo "    Follow logs with: tail -f \"$TRAIN_LOG\""
else
  echo "[$(date '+%F %T')] Warning: training process exited immediately. Inspect $TRAIN_LOG." >&2
  exit 1
fi
