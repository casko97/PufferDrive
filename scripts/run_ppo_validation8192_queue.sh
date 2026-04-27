#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_ppo_validation8192_queue_logs"
mkdir -p "$LOG_DIR"

WARMSTART_CKPT="experiments_warmstart_large/puffer_drive_177722345679/model_puffer_drive_000008.pt"

configs=(
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_validation8192_conservative.ini"
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_validation8192_std015.ini"
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_validation8192_lr1e4.ini"
)

tags=(
  "ppo_validation8192_conservative_from_large_ckpt8_20260426"
  "ppo_validation8192_std015_from_large_ckpt8_20260426"
  "ppo_validation8192_lr1e4_from_large_ckpt8_20260426"
)

for (( idx=1; idx<=${#configs[@]}; idx++ )); do
  config="${configs[$idx]}"
  tag="${tags[$idx]}"
  log_path="$LOG_DIR/${tag}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting $tag with $config" | tee -a "$log_path"
  PYTHONPATH=. .venv/bin/python scripts/run_packaged_drive_train.py \
    --config "$config" \
    --load-model-path "$WARMSTART_CKPT" \
    --tag "$tag" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished $tag" | tee -a "$log_path"
done
