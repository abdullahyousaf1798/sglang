#!/usr/bin/env bash
# Final table (REVIEW.md, revised design) on one pod: 1-minute windows at slowdown 15
# (under capacity) for both pools, slowdown 11 (saturation) on the pinned pool, repeats
# of the key rows, and the conversation trace. Windows are given explicitly so the
# work can be split across pods; each window's rows run on one machine.
#   nohup bash cache_aware_mlfq/run_review_final.sh "<natural windows>" "<pinned windows>" [conversation-window] &
set -uo pipefail
natural=$1 pinned=$2 conv=${3:-}
source /workspace/env.sh
cd /opt/sglang
R=/workspace/results
rm -f /workspace/keep_running "/workspace/keep_running.${RUNPOD_POD_ID:-none}"
RUN="bash cache_aware_mlfq/run_mooncake_main.sh"
ROWS="fcfs lpm shortest-prefill-first fcfs-interval8 lpm-mixed8k ours-final ours-final-yield fastserve-final"
KEY="fcfs lpm-mixed8k shortest-prefill-first ours-final"
[ -n "$pinned" ] && POOL=160000 TAG=m1-sd15 CONFIGS="$ROWS" $RUN toolagent 15 1.0 $pinned
[ -n "$natural" ] && TAG=m1-sd15 CONFIGS="$ROWS" $RUN toolagent 15 1.0 $natural
[ -n "$pinned" ] && POOL=160000 TAG=m1 CONFIGS="$KEY" $RUN toolagent 11 1.0 $pinned
[ -n "$pinned" ] && POOL=160000 TAG=m1-sd15-r2 CONFIGS="fcfs ours-final" $RUN toolagent 15 1.0 $pinned
[ -n "$conv" ] && TAG=m1-sd15 CONFIGS="fcfs lpm-mixed8k ours-final" $RUN conversation 15 1.0 $conv
date -u > "$R/.review_final_done.${RUNPOD_POD_ID:-$(hostname)}"
echo "final queue done: $(date -u)"
