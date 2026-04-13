#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/casko/phd-code/PufferDrive"
DEFAULT_INI="$ROOT/pufferlib/config/ocean/drive.ini"
BACKUP_INI="/tmp/drive.ini.backup.bc_streaming_full_default_20260317_134400_eval"
CHECKPOINT="$ROOT/pufferlib/resources/drive/models/bc-streaming-full-default-20260317-134400/puffer_drive_bc_streaming_full_default_20260317_134400.pt"
SCRIPT="$ROOT/scripts/run_packaged_drive_human_replay_eval.py"
PYTHON="$ROOT/pufferdrive.venv/bin/python3"

cleanup() {
    if [[ -f "$BACKUP_INI" ]]; then
        cp "$BACKUP_INI" "$DEFAULT_INI"
    fi
}

trap cleanup EXIT

cp "$DEFAULT_INI" "$BACKUP_INI"

ensure_output_dir_is_clean() {
    local output_dir="$1"
    local existing

    existing=$(find "$output_dir" -maxdepth 1 -type f \
        \( -name 'human_replay_eval_aggregate.json' \
        -o -name 'human_replay_eval_per_scenario.csv' \
        -o -name 'human_replay_eval_progress.json' \
        -o -name 'human_replay_eval_run.log' \
        -o -name 'human_replay_eval_sampled_maps.json' \
        -o -name 'human_replay_eval_skipped_scenarios.json' \))

    if [[ -n "$existing" ]]; then
        echo "Refusing to overwrite existing eval artifacts in: $output_dir" >&2
        echo "$existing" >&2
        exit 1
    fi
}

run_eval() {
    local config="$1"
    local output_dir="$2"

    ensure_output_dir_is_clean "$output_dir"

    echo "Running eval with config: $config"
    cp "$config" "$DEFAULT_INI"
    "$PYTHON" "$SCRIPT" \
        --config "$config" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$output_dir" \
        --num-envs 1 \
        --sample-size 100 \
        --sample-seed 42
}

run_eval \
    "$ROOT/pufferlib/resources/drive/models/bc-streaming-full-default-20260317-134400/car eval 2/conf1_discrete_human_replay_eval.ini" \
    "$ROOT/pufferlib/resources/drive/models/bc-streaming-full-default-20260317-134400/car eval 2"

run_eval \
    "$ROOT/pufferlib/resources/drive/models/bc-streaming-full-default-20260317-134400/truck eval 3/conf1_discrete_human_replay_eval.ini" \
    "$ROOT/pufferlib/resources/drive/models/bc-streaming-full-default-20260317-134400/truck eval 3"
