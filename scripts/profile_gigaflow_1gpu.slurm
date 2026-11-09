#!/usr/bin/env bash
# Submit with: sbatch -A <project-id> scripts/profile_gigaflow_1gpu.slurm
#
# This deliberately uses the same puffer_drive_gigaflow configuration as the
# 8-GPU job. Its one rank therefore has the same local rollout/batch geometry:
# 250 agents, 128 steps, and a 32,000-sample PPO update. It omits only DDP's
# cross-GPU gradient synchronization, so it is suitable for assessing per-GPU
# simulator, inference, and optimization load (but not NCCL overhead).
#
# The one-hour limit is intentional: this is a load-profiling run, not a full
# training job. Increase it if a longer measurement window is useful.
#SBATCH --job-name=pufferdrive-giga-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate puffer-nov.venv

cd "${SLURM_SUBMIT_DIR:?Submit this job from the PufferDrive repository root}"
mkdir -p experiments logs

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Export it before submitting." >&2
  exit 1
fi

test -f resources/drive/binaries/training/map_000.bin

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Keep W&B's local files and PufferLib checkpoints together under the Slurm ID.
wandb_results_dir="${PWD}/logs/wandb-results-${SLURM_JOB_ID}"
mkdir -p "${wandb_results_dir}"
export WANDB_DIR="${wandb_results_dir}"
wandb_group="${WANDB_GROUP:-gigaflow-1gpu-profile}-job-${SLURM_JOB_ID}"
wandb_name="${WANDB_RUN_NAME:-gigaflow-1gpu-profile}-job-${SLURM_JOB_ID}"

# Sample utilization, memory, clocks, power, and temperature every five
# seconds. The resulting CSV remains available after Slurm ends the run.
gpu_log="logs/gigaflow-1gpu-${SLURM_JOB_ID}.csv"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "${gpu_log}" &
nvidia_smi_pid=$!
trap 'kill "${nvidia_smi_pid}" 2>/dev/null || true' EXIT

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  -m pufferlib.pufferl train puffer_drive_gigaflow \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-pufferdrive}" \
  --wandb-group "${wandb_group}" \
  --wandb-name "${wandb_name}" \
  --train.data-dir "${wandb_results_dir}"
