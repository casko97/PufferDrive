#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/make_trailer_video.sh --bin <map_000.bin> --output <video.mp4> [options]

Options:
  --bin PATH             Input .bin scenario file (required)
  --output PATH          Output .mp4 path (required)
  --start N              First frame index (default: 0)
  --end N                Last frame index, inclusive (default: 90)
  --fps N                Output video FPS (default: 10)
  --workdir PATH         Directory for intermediate PNG frames (default: temporary dir)
  --python PATH          Python interpreter (default: ./pufferdrive.venv/bin/python if present, else python)
  --no-other-objects     Only draw tractor/trailer + roads
  -h, --help             Show this help
EOF
}

BIN_PATH=""
OUTPUT_PATH=""
START_FRAME=0
END_FRAME=90
FPS=10
WORKDIR=""
PYTHON_BIN=""
NO_OTHER_OBJECTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      BIN_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --start)
      START_FRAME="$2"
      shift 2
      ;;
    --end)
      END_FRAME="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-other-objects)
      NO_OTHER_OBJECTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$BIN_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "--bin and --output are required." >&2
  usage
  exit 2
fi

if [[ ! -f "$BIN_PATH" ]]; then
  echo "Input .bin not found: $BIN_PATH" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH." >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "./pufferdrive.venv/bin/python" ]]; then
    PYTHON_BIN="./pufferdrive.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d /tmp/trailer_video.XXXXXX)"
  CLEANUP_WORKDIR=1
else
  mkdir -p "$WORKDIR"
  CLEANUP_WORKDIR=0
fi

cleanup() {
  if [[ "${CLEANUP_WORKDIR:-0}" -eq 1 ]]; then
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

if [[ "$END_FRAME" -lt "$START_FRAME" ]]; then
  echo "--end must be >= --start" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

FRAME_LIST="$(seq -s, "$START_FRAME" "$END_FRAME")"
FRAME_PREFIX="$WORKDIR/frame.png"

VIZ_ARGS=(
  scripts/visualize_trailer_scene.py
  --bin "$BIN_PATH"
  --output "$FRAME_PREFIX"
  --frames "$FRAME_LIST"
)

if [[ "$NO_OTHER_OBJECTS" -eq 1 ]]; then
  VIZ_ARGS+=(--no-other-objects)
fi

echo "Rendering frames to: $WORKDIR"
"$PYTHON_BIN" "${VIZ_ARGS[@]}"

echo "Encoding video: $OUTPUT_PATH"
ffmpeg -y \
  -framerate "$FPS" \
  -i "$WORKDIR/frame_t%03d.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  "$OUTPUT_PATH" >/dev/null 2>&1

echo "Done: $OUTPUT_PATH"
if [[ "$CLEANUP_WORKDIR" -eq 0 ]]; then
  echo "Frames kept in: $WORKDIR"
fi

