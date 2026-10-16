#!/usr/bin/env bash
# Copy benchmark results from the pod's volume into the Mac repo.
set -euo pipefail
HOST=${POD_HOST:-runpod-sglang}
DEST=$(cd "$(dirname "$0")" && pwd)/results
mkdir -p "$DEST"
rsync -az "$HOST:/workspace/results/" "$DEST/"
echo "pulled results -> $DEST"
