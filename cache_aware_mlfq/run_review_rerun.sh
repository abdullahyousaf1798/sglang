#!/usr/bin/env bash
# Rerun rows that crashed, after this pod's own queue has finished (its done marker).
#   nohup bash /workspace/scripts/run_review_rerun.sh <marker-file> &
set -uo pipefail
marker=$1
source /workspace/env.sh
cd /opt/sglang
pod=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p"); pod=${pod:-$(hostname)}
touch "/workspace/keep_running.$pod"
until [ -f "$marker" ]; do sleep 30; done
RUN="bash cache_aware_mlfq/run_mooncake_main.sh"
# Lost to the chunk over-commit (shortest-prefill-first) and to the host-memory race (interval8).
POOL=160000 TAG=m1-sd15 CONFIGS="shortest-prefill-first" $RUN toolagent 15 1.0 25 55
POOL=160000 TAG=m1-sd15 CONFIGS="fcfs-interval8" $RUN toolagent 15 1.0 55
rm -f "/workspace/keep_running.$pod"
date -u > "/workspace/results/.rerun_done.$pod"
echo "rerun done: $(date -u)"
