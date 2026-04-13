#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/casko/phd-code/PufferDrive"
DEFAULT_INI="$ROOT/pufferlib/config/ocean/drive.ini"
BACKUP_INI="/tmp/drive.ini.backup.aug_car_baseline_full_boston_split_eval"
CHECKPOINT="$ROOT/resources/drive/models/aug-car-baseline-full-boston-split/puffer_drive_vjy06o6x.pt"
SCRIPT="$ROOT/scripts/run_packaged_drive_human_replay_eval.py"
PYTHON="$ROOT/pufferdrive.venv/bin/python3"

cleanup() {
    if [[ -f "$BACKUP_INI" ]]; then
        cp "$BACKUP_INI" "$DEFAULT_INI"
    fi
}

trap cleanup EXIT

cp "$DEFAULT_INI" "$BACKUP_INI"

run_eval() {
    local config="$1"
    local output_dir="$2"

    echo "Running eval with config: $config"
    cp "$config" "$DEFAULT_INI"
    awk 'BEGIN{in_base=0} /^\[/{in_base=(tolower($0)=="[base]")} {if(in_base && $0 ~ /^load_model_path[[:space:]]*=/) next; print}' "$DEFAULT_INI" > "${DEFAULT_INI}.tmp"
    mv "${DEFAULT_INI}.tmp" "$DEFAULT_INI"
    "$PYTHON" "$SCRIPT" \
        --config "$config" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$output_dir" \
        --num-envs 1 \
        --sample-size 100 \
        --sample-seed 42
}

run_eval \
    "$ROOT/resources/drive/models/aug-car-baseline-full-boston-split/car eval boston validation/conf1_discrete_human_replay_eval.ini" \
    "$ROOT/resources/drive/models/aug-car-baseline-full-boston-split/car eval boston validation"

run_eval \
    "$ROOT/resources/drive/models/aug-car-baseline-full-boston-split/truck eval boston validation/conf1_discrete_human_replay_eval.ini" \
    "$ROOT/resources/drive/models/aug-car-baseline-full-boston-split/truck eval boston validation"
