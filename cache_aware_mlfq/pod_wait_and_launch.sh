#!/usr/bin/env bash
# Pod side, shared network volume: wait for the runtime snapshot another pod is
# building, restore it, then run this pod's slice of the final table.
#   nohup bash /workspace/scripts/pod_wait_and_launch.sh "<natural windows>" "<pinned windows>" [conversation-window] &
set -uo pipefail
natural=$1 pinned=$2 conv=${3:-}
pod=${RUNPOD_POD_ID:-$(hostname)}
until [ -s /workspace/snapshots/runtime.tar.zst ] && [ -f /workspace/snapshots/BUILD_DONE ]; do sleep 30; done
bash /workspace/scripts/pod_resume.sh
# The Mac syncs the branch on top of the base checkout; wait for its marker.
until [ -f "/workspace/sync_done.$pod" ]; do sleep 20; done
cd /opt/sglang
bash cache_aware_mlfq/run_review_final.sh "$natural" "$pinned" "$conv" > "/workspace/logs/final_$pod.log" 2>&1
date -u > "/workspace/results/.final_done.$pod"
