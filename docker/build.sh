#!/usr/bin/env bash
# Build both endo-sam and endo-droid images.
# Usage: ./build.sh [sam|droid|all]   (default: all)
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-all}"

build_one() {
    local name="$1" dockerfile="$2"
    echo ""
    echo "==> Building image: $name ($dockerfile)"
    docker build -t "$name" -f "$dockerfile" .
}

case "$TARGET" in
    sam)   build_one endo-sam   Dockerfile.sam ;;
    droid) build_one endo-droid Dockerfile.droid ;;
    all)
        build_one endo-sam   Dockerfile.sam
        build_one endo-droid Dockerfile.droid
        ;;
    *)
        echo "Usage: $0 [sam|droid|all]"
        exit 1
        ;;
esac

echo ""
echo "==> Done. Current images:"
docker images | grep -E '^(endo-sam|endo-droid)\b' || true
