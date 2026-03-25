#!/usr/bin/env bash
set -u

cd /home/casko/phd-code/PufferDrive || exit 1

OUT="/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/apricot-aug-car-model-c5e2cdlo/truck eval 3/videos_worst_zoom06"
CFG="/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/apricot-aug-car-model-c5e2cdlo/truck eval 3/conf1_discrete_human_replay_eval.ini"
POL="/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/models/apricot-aug-car-model-c5e2cdlo/puffer_drive_weights.bin"
DRIVE_INI="/home/casko/phd-code/PufferDrive/pufferlib/config/ocean/drive.ini"
MAP_DIR="/home/casko/phd-code/PufferDrive/pufferlib/resources/drive/binaries/validation"

mkdir -p "$OUT"

for MAP in \
  map_3036.bin \
  map_1555.bin \
  map_657.bin \
  map_1005.bin \
  map_2203.bin \
  map_4165.bin \
  map_4233.bin \
  map_8895.bin \
  map_2097.bin \
  map_656.bin
do
  /home/casko/phd-code/PufferDrive/pufferdrive.venv/bin/python3 -c "from shutil import copy2; copy2(r'$CFG', r'$DRIVE_INI')"
  BASE="${MAP%.bin}"
  echo "rendering $MAP"
  xvfb-run -a -s "-screen 0 1280x720x24" ./visualize \
    --policy-name "$POL" \
    --map-name "$MAP_DIR/$MAP" \
    --log-trajectories \
    --zoom-in \
    --zoom-scale 0.6 \
    --view topdown \
    --output-topdown "$OUT/${BASE}_topdown.mp4" \
    --output-agent "$OUT/${BASE}_agent.mp4" \
    > "/tmp/${BASE}_augcar_eval3_render.log" 2>&1 || true
done
