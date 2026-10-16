#!/usr/bin/env bash
# Copy benchmark results from the pod's volume into the Mac repo.
set -euo pipefail
HOST=${POD_HOST:-runpod-sglang}
DEST=$(cd "$(dirname "$0")" && pwd)/results
mkdir -p "$DEST"
# Profiles live one level up (cache_aware_mlfq/prefill_profile_*.json), copied by hand.
rsync -az --exclude "prefill_profile_*.json" "$HOST:/workspace/results/" "$DEST/"
echo "pulled results -> $DEST"
