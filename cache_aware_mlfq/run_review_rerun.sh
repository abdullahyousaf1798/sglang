#!/usr/bin/env bash
# Rerun pinned slowdown-15 rows that crashed, after this pod's own queue has finished.
# The crashed row's folder is kept as <name>.crashed-<n> (its server.log is the evidence).
#   nohup bash /workspace/scripts/run_review_rerun.sh <marker-file> <config:window>... &
set -uo pipefail
marker=$1; shift
source /workspace/env.sh
cd /opt/sglang
R=/workspace/results
pod=$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p"); pod=${pod:-$(hostname)}
touch "/workspace/keep_running.$pod"
until [ -f "$marker" ]; do sleep 30; done
for item in "$@"; do
  config=${item%:*} window=${item##*:}
  name="mooncake160k-toolagent-$config-s$window-m1-sd15"
  if [ -d "$R/$name" ] && [ ! -f "$R/$name/summary.json" ]; then
    n=1; while [ -e "$R/$name.crashed-$n" ]; do n=$((n + 1)); done
    mv "$R/$name" "$R/$name.crashed-$n"
  fi
  [ -f "$R/$name/summary.json" ] && { echo "skip $name: already complete"; continue; }
  POOL=160000 TAG=m1-sd15 CONFIGS="$config" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 15 1.0 "$window"
done
rm -f "/workspace/keep_running.$pod"
date -u > "$R/.rerun_done.$pod"
echo "rerun done: $(date -u)"
