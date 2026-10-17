#!/usr/bin/env bash
# Main table on Mooncake production traces (A100 80 GB): every configuration on each
# given trace window, window-major, so each window's rows run back to back on one pod.
# Runs on the pod in the background; writes a marker when every row has finished.
#
#   bash cache_aware_mlfq/run_mooncake_main.sh <trace: toolagent|conversation> \
#       <slowdown> <window-minutes> <window-start-minute>...
set -uo pipefail
source /workspace/env.sh
trace=$1 slowdown=$2 minutes=$3
shift 3
marker=/workspace/results/.mooncake_main_done
rm -f "$marker"
PROFILE=/opt/sglang/cache_aware_mlfq/prefill_profile_a100-sxm4-80gb.json
MLFQ=(
  --schedule-policy sj-mlfq
  --sj-mlfq-profile "$PROFILE"
  --enable-priority-scheduling
  --priority-scheduling-preemption-threshold 0
  --retraction-policy priority
)
# FastServe's mechanisms on SGLang: preempted KV is kept, evicted lowest level first
# after finished conversations (4b), and loaded back ahead of admission (5).
MECHANISMS=(--preempt-keep-kv --radix-eviction-policy mlfq --sj-mlfq-load-ahead)
row() {
  local name=$1 start=$2
  shift 2
  python cache_aware_mlfq/run_experiment.py --name "mooncake-$trace-$name-s$start" \
    --workload mooncake --trace "/workspace/data/mooncake_$trace.jsonl" \
    --start-min "$start" --minutes "$minutes" --slowdown "$slowdown" -- "$@" \
    || echo "FAILED: $name window $start"
}
for start in "$@"; do
  row fcfs "$start" --schedule-policy fcfs
  row lpm "$start" --schedule-policy lpm
  row shortest-prefill-first "$start" --schedule-policy shortest-prefill-first
  row sj-mlfq-naive "$start" "${MLFQ[@]}" --sj-mlfq-cost prompt
  row fastserve "$start" "${MLFQ[@]}" --sj-mlfq-cost prompt "${MECHANISMS[@]}"
  row ours "$start" "${MLFQ[@]}" --sj-mlfq-cost cache-aware "${MECHANISMS[@]}"
  row ours-no-load-ahead "$start" "${MLFQ[@]}" --sj-mlfq-cost cache-aware \
    --preempt-keep-kv --radix-eviction-policy mlfq
  row ours-lru-eviction "$start" "${MLFQ[@]}" --sj-mlfq-cost cache-aware \
    --preempt-keep-kv --sj-mlfq-load-ahead
done
date -u > "$marker"
