#!/usr/bin/env bash
# Launch a pipeline stage inside the right Docker image.
#
# Usage:
#   ./docker/run.sh sam   <run.py subcommand> [args...]
#   ./docker/run.sh droid <run.py subcommand> [args...]
#   ./docker/run.sh sam   shell                     # interactive bash inside the image
#   ./docker/run.sh droid shell
#
# Examples:
#   ./docker/run.sh sam   sam2-propagate --out ./out --ckpt /models/sam2.1_hiera_large.pt
#   ./docker/run.sh droid droid           --out ./out --stride 1
#
# Volume mounts:
#   - The whole project (parent of this script) is mounted at /workspace
#   - $MODELS_DIR (default ~/models) is mounted read-only at /models
#   - Output files written under ./out land in your project dir on the host.

set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \?//'
    exit 1
fi

KIND="$1"; shift

case "$KIND" in
    sam)   IMAGE="endo-sam" ;;
    droid) IMAGE="endo-droid" ;;
    *) echo "Unknown kind: $KIND (expected sam or droid)"; exit 1 ;;
esac

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# Allow shell mode
if [ "${1:-}" = "shell" ]; then
    CMD=(bash)
else
    CMD=(python run.py "$@")
fi

# Use host UID/GID so files written under ./out are owned by you, not root.
# --gpus all gives container all GPUs visible to the host (needs nvidia-container-toolkit).
echo "==> Running in container: $IMAGE"
echo "    project: $PROJECT_DIR  →  /workspace"
echo "    models:  $MODELS_DIR   →  /models  (read-only)"
echo "    cmd:     ${CMD[*]}"
echo ""

mkdir -p "$MODELS_DIR"

docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --shm-size=8g \
    -v "$PROJECT_DIR":/workspace \
    -v "$MODELS_DIR":/models:ro \
    -w /workspace \
    "$IMAGE" \
    "${CMD[@]}"
