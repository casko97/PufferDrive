#!/usr/bin/env bash
set -euo pipefail

cd /home/casko/phd-code/PufferDrive

PY="/home/casko/phd-code/PufferDrive/pufferdrive.venv/bin/python3"
RUNNER="/home/casko/phd-code/PufferDrive/scripts/run_packaged_drive_human_replay_eval.py"
DRIVE_INI="/home/casko/phd-code/PufferDrive/pufferlib/config/ocean/drive.ini"

run_eval() {
  local cfg="$1"
  local ckpt="$2"
  local out_dir="$3"

  "$PY" -c "from shutil import copy2; copy2(r'$cfg', r'$DRIVE_INI'); print('copied eval ini to drive.ini')"
  "$PY" "$RUNNER" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --sample-size 100 \
    --sample-seed 42 \
    > "$out_dir/human_replay_eval_run.log" 2>&1
}

run_eval \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/elated-capybara-125-dkmcai60-new-aug-truck-model/truck eval 3/conf1_discrete_human_replay_eval.ini" \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/elated-capybara-125-dkmcai60-new-aug-truck-model/puffer_drive_dkmcai60.pt" \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/elated-capybara-125-dkmcai60-new-aug-truck-model/truck eval 3"

run_eval \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/vivid-forest-124-3yu4idjo-c5e2cdlo-truck-finetuning/truck eval 3/conf1_discrete_human_replay_eval.ini" \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/vivid-forest-124-3yu4idjo-c5e2cdlo-truck-finetuning/puffer_drive_3yu4idjo.pt" \
  "/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/vivid-forest-124-3yu4idjo-c5e2cdlo-truck-finetuning/truck eval 3"
