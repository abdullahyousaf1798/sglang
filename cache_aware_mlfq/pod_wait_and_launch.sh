#!/usr/bin/env bash
# Pod side, shared network volume: wait for the runtime snapshot (built once, on one
# pod, with the branch already synced into it), restore it, then run this pod's
# slice of the final table.
#   nohup bash /workspace/scripts/pod_wait_and_launch.sh "<natural windows>" "<pinned windows>" [conversation-window] &
set -uo pipefail
natural=$1 pinned=$2 conv=${3:-}
pod=${RUNPOD_POD_ID:-$(hostname)}
touch "/workspace/keep_running.$pod"
until [ -f /workspace/snapshots/BUILD_DONE ]; do sleep 30; done
bash /workspace/scripts/pod_resume.sh
cd /opt/sglang
bash cache_aware_mlfq/run_review_final.sh "$natural" "$pinned" "$conv" > "/workspace/logs/final_$pod.log" 2>&1
date -u > "/workspace/results/.final_done.$pod"
