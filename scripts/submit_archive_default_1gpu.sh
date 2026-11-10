#!/usr/bin/env bash
# Submit the one-GPU archived-default PufferDrive run with mandatory W&B logging.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/proj/rpl-soro/users/x_carsk/pufferdrive-november}"
cd "${PROJECT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Export it before submitting." >&2
  exit 1
fi

mkdir -p logs
export WANDB_PROJECT="${WANDB_PROJECT:-pufferdrive}"
export WANDB_GROUP="${WANDB_GROUP:-archive-default-1gpu-$(date +%Y%m%d-%H%M%S)}"

echo "Submitting archived-default one-GPU run to W&B project: ${WANDB_PROJECT}"
echo "W&B group: ${WANDB_GROUP}"
sbatch --export=ALL scripts/train_archive_default_1gpu.slurm
