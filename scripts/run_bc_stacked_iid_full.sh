#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-stacked-iid-full-lr1e3-20260501_174200"
LOG_PATH="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"
exec >>"${LOG_PATH}" 2>&1

echo "[launcher] starting stacked_iid full BC run at $(date --iso-8601=seconds)"
echo "[launcher] log_path=${LOG_PATH}"

cd /tmp/PufferDrive-bc-worktree
export PYTHONPATH=/tmp/PufferDrive-bc-worktree
export PYTHONUNBUFFERED=1

/home/casko/phd-code/PufferDrive/.venv/bin/python3 /tmp/PufferDrive-bc-worktree/scripts/launch_bc_stacked_iid_train.py \
  --config /tmp/PufferDrive-bc-worktree/pufferlib/config/ocean/drive_bc_paired_offline_fits_stacked_iid_car.ini \
  --output-dir /home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-stacked-iid-full-lr1e3-20260501_174200 \
  --device cuda \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 2 \
  --shard-shuffle-buffer 4 \
  --max-maps -1 \
  --val-fraction 0.1 \
  --log-interval 25 \
  --index-log-interval 25 \
  --learning-rate 0.001
