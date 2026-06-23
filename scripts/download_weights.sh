#!/usr/bin/env bash
# Download SAM 2 weights into ./models (or $MODELS_DIR).
# DROID-SLAM weight is already baked into the endo-droid docker image
# (see docker/Dockerfile.droid), so it is NOT downloaded here.
#
# Usage:
#   bash scripts/download_weights.sh
#   MODELS_DIR=~/models bash scripts/download_weights.sh

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_OUT="$MODELS_DIR/sam2.1_hiera_large.pt"

if [[ -f "$SAM2_OUT" ]]; then
    echo "[skip] $SAM2_OUT already present ($(du -h "$SAM2_OUT" | cut -f1))"
else
    echo "[get ] $SAM2_URL"
    wget -c "$SAM2_URL" -O "$SAM2_OUT"
fi

echo
echo "Done. Models in $MODELS_DIR/:"
ls -lh "$MODELS_DIR"
