#!/usr/bin/env bash
# Main table on the realistic workload: real multi-turn ShareGPT conversations,
# three seeds per configuration (the same seeds for every configuration, so rows
# compare like for like). Seed-major order keeps partial results balanced.
# Runs on the pod in the background; writes a marker when every row has finished.
#
#   bash cache_aware_mlfq/run_sharegpt_main.sh <request-rate> <num-conversations>
set -uo pipefail
source /workspace/env.sh
rate=$1
conversations=$2
marker=/workspace/results/.sharegpt_main_done
rm -f "$marker"
MLFQ=(
  --schedule-policy sj-mlfq
  --sj-mlfq-profile /opt/sglang/cache_aware_mlfq/prefill_profile_4090.json
  --enable-priority-scheduling
  --priority-scheduling-preemption-threshold 0
  --retraction-policy priority
)
row() {
  local name=$1 seed=$2
  shift 2
  python cache_aware_mlfq/run_experiment.py --name "sharegpt-$name-s$seed" --workload sharegpt \
    --request-rate "$rate" --num-conversations "$conversations" --seed "$seed" -- "$@" \
    || echo "FAILED: $name seed $seed"
}
for seed in 1 2 3; do
  row fcfs "$seed" --schedule-policy fcfs
  row lpm "$seed" --schedule-policy lpm
  row shortest-prefill-first "$seed" --schedule-policy shortest-prefill-first
  row sj-mlfq-prompt "$seed" "${MLFQ[@]}" --sj-mlfq-cost prompt
  row sj-mlfq-cache-aware "$seed" "${MLFQ[@]}" --sj-mlfq-cost cache-aware
  row sj-mlfq-cache-aware-keepkv "$seed" "${MLFQ[@]}" --sj-mlfq-cost cache-aware --preempt-keep-kv
done
date -u > "$marker"
