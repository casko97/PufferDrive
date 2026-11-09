#!/usr/bin/env bash
# Submit a GIGAFLOW-inspired PufferDrive job with mandatory W&B logging.
# Usage: scripts/submit_gigaflow.sh [train|profile]

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/proj/rpl-soro/users/x_carsk/pufferdrive-november}"
cd "${PROJECT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Export it before submitting." >&2
  exit 1
fi

run_type="${1:-train}"
case "${run_type}" in
  train)
    job_script="scripts/train_gigaflow.slurm"
    default_group="gigaflow-inspired-$(date +%Y%m%d-%H%M%S)"
    ;;
  profile)
    job_script="scripts/profile_gigaflow_1gpu.slurm"
    default_group="gigaflow-1gpu-profile-$(date +%Y%m%d-%H%M%S)"
    ;;
  *)
    echo "Usage: $0 [train|profile]" >&2
    exit 2
    ;;
esac

mkdir -p logs
export WANDB_PROJECT="${WANDB_PROJECT:-pufferdrive}"
export WANDB_GROUP="${WANDB_GROUP:-${default_group}}"

echo "Submitting ${run_type} run to W&B project: ${WANDB_PROJECT}"
echo "W&B group: ${WANDB_GROUP}"
sbatch --export=ALL "${job_script}"
