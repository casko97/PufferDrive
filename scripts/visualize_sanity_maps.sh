#!/usr/bin/env bash
set -euo pipefail

# Repo-root relative defaults
POLICY_PATH_DEFAULT="pufferlib/resources/drive/models/apricot-voice-35/puffer_drive_weights.bin"
MAP_DIR_DEFAULT="pufferlib/resources/drive/binaries/sanity"
OUT_TOPDOWN_DIR_DEFAULT="pufferlib/resources/drive/models/apricot-voice-35/videos"
OUT_AGENT_DIR_DEFAULT="pufferlib/resources/drive/models/apricot-voice-35/videos"

# Allow overrides via env vars
POLICY_PATH="${POLICY_PATH_OVERRIDE:-$POLICY_PATH_DEFAULT}"
MAP_DIR="${MAP_DIR_OVERRIDE:-$MAP_DIR_DEFAULT}"
OUT_TOPDOWN_DIR="${OUT_TOPDOWN_DIR_OVERRIDE:-$OUT_TOPDOWN_DIR_DEFAULT}"
OUT_AGENT_DIR="${OUT_AGENT_DIR_OVERRIDE:-$OUT_AGENT_DIR_DEFAULT}"

usage() {
  cat <<'EOF'
Usage: scripts/visualize_sanity_maps.sh [options]

Options:
  --map-dir PATH       Directory containing map_*.bin (default: sanity dir)
  --policy PATH        Path to model weights/bin (default: apricot-voice-35)
  --topdown-dir PATH   Output directory for topdown videos
  --agent-dir PATH     Output directory for agent videos
  --sample N           Process every Nth map (default: 1, i.e. all maps)
  -h, --help           Show this help and exit

Env overrides:
  POLICY_PATH_OVERRIDE, MAP_DIR_OVERRIDE, OUT_TOPDOWN_DIR_OVERRIDE, OUT_AGENT_DIR_OVERRIDE
EOF
}

SAMPLE_EVERY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map-dir)
      MAP_DIR="$2"
      shift 2
      ;;
    --policy)
      POLICY_PATH="$2"
      shift 2
      ;;
    --topdown-dir)
      OUT_TOPDOWN_DIR="$2"
      shift 2
      ;;
    --agent-dir)
      OUT_AGENT_DIR="$2"
      shift 2
      ;;
    --sample)
      SAMPLE_EVERY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x ./visualize ]]; then
  echo "ERROR: ./visualize not found or not executable. Run from repo root." >&2
  exit 1
fi

MAP_DIR_BASE="$(basename "$MAP_DIR")"

OUT_TOPDOWN_DIR="$OUT_TOPDOWN_DIR/$MAP_DIR_BASE"
OUT_AGENT_DIR="$OUT_AGENT_DIR/$MAP_DIR_BASE"

mkdir -p "$OUT_TOPDOWN_DIR" "$OUT_AGENT_DIR"

# Start Xvfb if not already running on :99
XVFB_PID=""
if ! pgrep -f "Xvfb :99" >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1280x720x24 &
  XVFB_PID=$!
  # Give Xvfb a moment to start
  sleep 0.5
fi

export DISPLAY=:99

cleanup() {
  if [[ -n "$XVFB_PID" ]]; then
    kill "$XVFB_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

i=0
for map in "$MAP_DIR"/map_*.bin; do
  [[ -e "$map" ]] || continue
  i=$((i + 1))
  base="$(basename "$map")"
  num="${base#map_}"
  num="${num%.bin}"

  # Normalize to 3 digits if numeric
  if [[ "$num" =~ ^[0-9]+$ ]]; then
    # Force base-10 to avoid octal interpretation (e.g., 008)
    num_dec=$((10#$num))
    num_fmt=$(printf "%03d" "$num_dec")
  else
    num_fmt="$num"
  fi

  # Optional sampling: only process every Nth map by index (1-based)
  if [[ "$SAMPLE_EVERY" =~ ^[0-9]+$ ]]; then
    if (( SAMPLE_EVERY > 1 )) && (( i % SAMPLE_EVERY != 0 )); then
      continue
    fi
  fi

  out_topdown="$OUT_TOPDOWN_DIR/topdown_${MAP_DIR_BASE}${num_fmt}.mp4"
  out_agent="$OUT_AGENT_DIR/agent_${MAP_DIR_BASE}${num_fmt}.mp4"

  echo "Visualizing $map -> $out_topdown + $out_agent"
  ./visualize \
    --policy-name "$POLICY_PATH" \
    --map-name "$map" \
    --log-trajectories \
    --zoom-in \
    --view topdown \
    --output-topdown "$out_topdown" \
    --output-agent "$out_agent"

done

echo "Done."
