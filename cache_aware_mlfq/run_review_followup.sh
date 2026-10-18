#!/usr/bin/env bash
# Continue a pod whose pinned slowdown-15 phase is already running under
# run_mooncake_main.sh: wait for it, then run the saturation and repeat rows for
# that window. Used when extra pods took over the other phases mid-run.
#   nohup bash /workspace/scripts/run_review_followup.sh <pinned window> &
set -uo pipefail
pinned=$1
source /workspace/env.sh
cd /opt/sglang
R=/workspace/results
while pgrep -f "^bash cache_aware_mlfq/run_mooncake_main.sh" >/dev/null; do sleep 30; done
RUN="bash cache_aware_mlfq/run_mooncake_main.sh"
KEY="fcfs lpm-mixed8k shortest-prefill-first ours-final"
POOL=160000 TAG=m1 CONFIGS="$KEY" $RUN toolagent 11 1.0 $pinned
POOL=160000 TAG=m1-sd15-r2 CONFIGS="fcfs ours-final" $RUN toolagent 15 1.0 $pinned
pod=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p"); pod=${pod:-$(hostname)}
rm -f "/workspace/keep_running.$pod"
date -u > "$R/.review_final_done.$pod"
echo "follow-up done: $(date -u)"
