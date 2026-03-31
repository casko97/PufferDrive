#!/bin/bash
# SLURM batch job pufferdrive_nuplan_waymo

#SBATCH --job-name=pufferdrive_nuplan_waymo
#SBATCH --account=berzelius-2026-36
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --constraint=thin
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=casko@kth.se

#SBATCH --output /proj/rpl-soro/users/x_carsk/PufferDrive/logs/outlog-%J.log
#SBATCH --error  /proj/rpl-soro/users/x_carsk/PufferDrive/logs/errlog-%J.log

set -eo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

log "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# Load environment
log "Loading modules"
module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0
log "Activating conda environment: pufferdrive_tractorTrailer"
mamba activate pufferdrive_tractorTrailer

log "Environment activated, starting setup"

# Insall and setup
REPO_DIR=/proj/rpl-soro/users/x_carsk/PufferDrive/PufferDrive
ACTIVE_DRIVE_INI="${REPO_DIR}/pufferlib/config/ocean/drive.ini"
log "Changing to repository directory: ${REPO_DIR}"
cd "${REPO_DIR}"
#./conda_install.sh
#./scripts/build_ocean.sh visualize local
#export CC=gcc
export WANDB_API_KEY=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
log "Logging in to Weights & Biases"
wandb login

log "Setup complete, preparing training runs"

# Training batch configuration.
# This pre-configured INI file will be copied to pufferlib/config/ocean/drive.ini
# before each run, then its seed will be updated for the current launch.
SOURCE_DRIVE_INI="/proj/rpl-soro/users/x_carsk/PufferDrive/PufferDrive/large_scale_training.ini"

# Seed sweep settings for each config above. This launches NUM_SEED_RUNS runs
# per config with seeds START_SEED, START_SEED+1, ...
START_SEED=42
NUM_SEED_RUNS=1

WANDB_PROJECT="pufferdrive"
WANDB_GROUP_PREFIX="nuplam_waymo"
LOAD_MODEL_PATH=""

if [ -z "${SOURCE_DRIVE_INI}" ]; then
  log "No SOURCE_DRIVE_INI configured. Edit slurm_train_pufferdrive.sh and set the path to your drive.ini file."
  exit 1
fi

if [ ! -f "${ACTIVE_DRIVE_INI}" ]; then
  log "Active drive.ini not found at ${ACTIVE_DRIVE_INI}"
  exit 1
fi

update_drive_seed() {
  local ini_path="$1"
  local seed="$2"

  log "Updating seed to ${seed} in ${ini_path}"
  python - "${ini_path}" "${seed}" <<'PY'
import configparser
import sys

ini_path = sys.argv[1]
seed = sys.argv[2]

parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
parser.optionxform = str
with open(ini_path, "r", encoding="utf-8") as f:
    parser.read_file(f)

for section in ("vec", "train"):
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, "seed", seed)

with open(ini_path, "w", encoding="utf-8") as f:
    parser.write(f)
PY
}

launch_training_run() {
  local seed="$1"

  local config_stem
  config_stem="$(basename "${SOURCE_DRIVE_INI}" .ini)"
  local run_name="${config_stem}-seed${seed}"
  local wandb_group="${WANDB_GROUP_PREFIX}-${config_stem}"

  log "----------------------------------------"
  log "Preparing run ${run_name}"
  log "Copying ${SOURCE_DRIVE_INI} -> ${ACTIVE_DRIVE_INI}"
  cp "${SOURCE_DRIVE_INI}" "${ACTIVE_DRIVE_INI}"
  update_drive_seed "${ACTIVE_DRIVE_INI}" "${seed}"
  log "Finished preparing config for ${run_name}"

  log "Launching training for ${run_name}"
  local cmd=(
    torchrun --standalone --nnodes=1 --nproc-per-node=4 -m pufferlib.pufferl train puffer_drive
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${wandb_group}"
  )

  if [ -n "${LOAD_MODEL_PATH}" ]; then
    log "Using model checkpoint: ${LOAD_MODEL_PATH}"
    cmd+=(--load-model-path "${LOAD_MODEL_PATH}")
  fi

  log "Command: ${cmd[*]}"
  "${cmd[@]}"
  log "Training run completed: ${run_name}"
}

# Run!
#pytest PufferDrive/tests/test_drive_bin_simulator_load.py:test_boston_tractor_trailer_setup
# torchrun --standalone --nnodes=1 --nproc-per-node=2 -m
if [ ! -f "${SOURCE_DRIVE_INI}" ]; then
  log "Configured INI not found: ${SOURCE_DRIVE_INI}"
  exit 1
fi

log "Using source config: ${SOURCE_DRIVE_INI}"
log "Active target config: ${ACTIVE_DRIVE_INI}"
log "Seed sweep starts at ${START_SEED} and will run ${NUM_SEED_RUNS} times"
log "Weights & Biases project: ${WANDB_PROJECT}"
log "Weights & Biases group prefix: ${WANDB_GROUP_PREFIX}"

for ((seed_offset=0; seed_offset<NUM_SEED_RUNS; seed_offset++)); do
  seed=$((START_SEED + seed_offset))
  log "Starting run $((seed_offset + 1)) of ${NUM_SEED_RUNS} with seed ${seed}"
  launch_training_run "${seed}"
done

log "All training runs completed. Script finalized."
