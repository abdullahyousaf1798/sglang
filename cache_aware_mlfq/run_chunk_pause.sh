#!/usr/bin/env bash
# Chunk pause row (ours-final-pause, DESIGN.md 6.1) on one pod: pinned 1-minute windows
# at slowdown 15, then natural-pool windows. Windows are given explicitly so the work
# can be split across pods; run_experiment.py refuses a row folder another pod already
# created, so overlapping window lists only cost a skipped row.
#   nohup setsid bash cache_aware_mlfq/run_chunk_pause.sh "<pinned windows>" "<natural windows>" \
#       > /workspace/logs/chunk_pause_<pod>.log 2>&1 < /dev/null &
set -uo pipefail
pinned=$1 natural=${2:-}
source /workspace/env.sh
cd /opt/sglang
pod=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p"); pod=${pod:-$(hostname)}
RUN="bash cache_aware_mlfq/run_mooncake_main.sh"
[ -n "$pinned" ] && POOL=160000 TAG=m1-sd15 CONFIGS=ours-final-pause $RUN toolagent 15 1.0 $pinned
[ -n "$natural" ] && TAG=m1-sd15 CONFIGS=ours-final-pause $RUN toolagent 15 1.0 $natural
date -u > "/workspace/results/.chunk_pause_done.$pod"
echo "chunk pause queue done: $(date -u)"
