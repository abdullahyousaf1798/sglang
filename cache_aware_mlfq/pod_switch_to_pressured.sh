#!/usr/bin/env bash
# One-off: once the natural-pool sweep has finished every row of its first window,
# stop it, add the prefill-gate rows for that window, then run the 160k-token
# GPU KV pool sweep on this pod's windows.
#
#   bash cache_aware_mlfq/pod_switch_to_pressured.sh <natural-window> <pressured-window>...
set -uo pipefail
source /workspace/env.sh
natural=$1
shift
last_row=/workspace/results/mooncake-toolagent-ours-lru-eviction-s$natural/summary.json
until [ -f "$last_row" ]; do sleep 20; done

# The bracket keeps pgrep from matching its own command line.
pkill -f "[r]un_mooncake_main.sh toolagent 11 0.4"
sleep 2
pkill -f "[r]un_experiment.py"
pkill -f "[m]ooncake_replay.py"
pkill -f "[s]glang.launch_server"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
# The row the stopped sweep had just started is incomplete.
for d in /workspace/results/mooncake-toolagent-*; do
  [ -f "$d/summary.json" ] || rm -rf "$d"
done

CONFIGS="fastserve-gate ours-gate" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$natural"
POOL=160000 CONFIGS="fcfs lpm shortest-prefill-first fastserve ours ours-no-load-ahead ours-lru-eviction fastserve-gate ours-gate" \
  bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$@"
