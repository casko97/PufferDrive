#!/usr/bin/env bash
# Submit with: scripts/submit_archive_default_1gpu.sh
#
# One-GPU run using pufferlib/config/archive/default.ini. The archived
# configuration keeps its original six Drive workers, 1,024 agents per worker,
# PPO hyperparameters, and recurrent policy. The only runtime overrides adapt
# its obsolete 10,000-map training/ directory to the 1,000 flat map binaries
# installed in this checkout and disable interactive rendering, which is not
# available in a headless Slurm allocation.
#SBATCH --job-name=pufferdrive-archive-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# The NSC Mamba wrapper reads CONDA_DEFAULT_ENV before it exists. Enable
# nounset only after activating the environment.
set -eo pipefail

module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0

mamba activate puffer-nov-cu121.venv
set -u

cd "${SLURM_SUBMIT_DIR:?Submit this job from the PufferDrive repository root}"
mkdir -p experiments logs

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Export it before submitting." >&2
  exit 1
fi

# archive/default.ini expects 10,000 maps in binaries/training. The available
# PufferDrive map set is flat and contains 1,000 maps, so make that explicit.
archive_map_dir="${ARCHIVE_MAP_DIR:-resources/drive/binaries}"
archive_num_maps="${ARCHIVE_NUM_MAPS:-1000}"
if [[ ! "${archive_num_maps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ARCHIVE_NUM_MAPS must be a positive integer" >&2
  exit 1
fi
if [[ ! -f "${archive_map_dir}/map_000.bin" ]]; then
  echo "Missing ${archive_map_dir}/map_000.bin" >&2
  exit 1
fi
available_maps=$(find "${archive_map_dir}" -maxdepth 1 -type f -name 'map_*.bin' | wc -l)
if (( available_maps < archive_num_maps )); then
  echo "ARCHIVE_NUM_MAPS=${archive_num_maps}, but only ${available_maps} map binaries are available in ${archive_map_dir}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Keep W&B files and PufferLib checkpoints isolated by Slurm job ID.
wandb_results_dir="${PWD}/logs/wandb-results-${SLURM_JOB_ID}"
mkdir -p "${wandb_results_dir}"
export WANDB_DIR="${wandb_results_dir}"
wandb_group="${WANDB_GROUP:-archive-default-1gpu}-job-${SLURM_JOB_ID}"
wandb_name="${WANDB_RUN_NAME:-archive-default-1gpu}-job-${SLURM_JOB_ID}"

# Persist GPU telemetry independently of PufferLib's dashboard. The latter
# does not currently report CUDA utilization correctly under torchrun.
gpu_log="logs/archive-default-1gpu-${SLURM_JOB_ID}.csv"
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
  -m pufferlib.pufferl train puffer_drive_archive_default \
  --env.map-dir "${archive_map_dir}" \
  --env.num-maps "${archive_num_maps}" \
  --train.render False \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-pufferdrive}" \
  --wandb-group "${wandb_group}" \
  --wandb-name "${wandb_name}" \
  --train.data-dir "${wandb_results_dir}"
