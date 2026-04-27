#!/bin/zsh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/experiments_ppo_ablation4096_refine_queue_logs"
mkdir -p "$LOG_DIR"

WARMSTART_CKPT="experiments_warmstart_large/puffer_drive_177722345679/model_puffer_drive_000008.pt"

configs=(
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_ablation4096_ent0.ini"
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_ablation4096_lr7e5.ini"
  "pufferlib/config/ocean/drive_trajectory_ppo_finetune_ablation4096_clip005.ini"
)

tags=(
  "ppo_ablation4096_ent0_from_large_ckpt8"
  "ppo_ablation4096_lr7e5_from_large_ckpt8"
  "ppo_ablation4096_clip005_from_large_ckpt8"
)

failures=0

for (( idx=1; idx<=${#configs[@]}; idx++ )); do
  config="${configs[$idx]}"
  tag="${tags[$idx]}"
  log_path="$LOG_DIR/${tag}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting $tag with $config" | tee -a "$log_path"
  PYTHONPATH=. .venv/bin/python scripts/run_packaged_drive_train.py \
    --config "$config" \
    --load-model-path "$WARMSTART_CKPT" \
    --tag "$tag" 2>&1 | tee -a "$log_path"
  exit_code=${pipestatus[1]}
  if [[ $exit_code -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] failed $tag exit_code=$exit_code" | tee -a "$log_path"
    failures=$((failures + 1))
    continue
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished $tag" | tee -a "$log_path"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue complete failures=$failures"
exit 0
