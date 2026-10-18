#!/usr/bin/env bash
# Pod side, shared network volume: wait for the runtime snapshot (built once, on one
# pod, with the branch already synced into it), restore it, then run this pod's
# slice of the final table. Arms the idle stop first and gives up after 2.5 h
# without a snapshot, so a failed build never leaves pods running all night.
#   nohup bash /workspace/scripts/pod_wait_and_launch.sh "<natural windows>" "<pinned windows>" [conversation-window] &
set -uo pipefail
natural=$1 pinned=$2 conv=${3:-}
# The pod id is in the container init environment, not in a non-login shell.
pod=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p"); pod=${pod:-$(hostname)}
mkdir -p /workspace/logs /workspace/results
pgrep -f "^bash /workspace/scripts/idle_stop.sh" >/dev/null \
  || (nohup setsid bash /workspace/scripts/idle_stop.sh >> "/workspace/logs/idle_stop_$pod.log" 2>&1 < /dev/null &)
touch "/workspace/keep_running.$pod"
for _ in $(seq 1 300); do [ -f /workspace/snapshots/BUILD_DONE ] && break; sleep 30; done
if [ ! -f /workspace/snapshots/BUILD_DONE ]; then
  echo "no runtime snapshot after 2.5 h; releasing the idle hold: $(date -u)"
  rm -f "/workspace/keep_running.$pod"
  exit 1
fi
bash /workspace/scripts/pod_resume.sh
cd /opt/sglang
bash cache_aware_mlfq/run_review_final.sh "$natural" "$pinned" "$conv" > "/workspace/logs/final_$pod.log" 2>&1
date -u > "/workspace/results/.final_done.$pod"
