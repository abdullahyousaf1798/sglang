#!/usr/bin/env bash
# Step 4 sweep: how much does sj-mlfq's churn depend on the preemption threshold
# and the base quantum, and how noisy are single runs? All MLFQ rows use
# cache-aware placement with preempted KV kept. Runs on the pod in the
# background; writes a marker when every row has finished.
set -uo pipefail
source /workspace/env.sh
marker=/workspace/results/.step4_sweep_done
rm -f "$marker"
MLFQ=(
  --schedule-policy sj-mlfq
  --sj-mlfq-profile /opt/sglang/cache_aware_mlfq/prefill_profile_4090.json
  --sj-mlfq-cost cache-aware
  --enable-priority-scheduling
  --retraction-policy priority
  --preempt-keep-kv
)
row() {
  local name=$1
  shift
  python cache_aware_mlfq/run_experiment.py --name "$name" -- "$@" || echo "FAILED: $name"
}
row sj-mlfq-cache-aware-keepkv-gap1 "${MLFQ[@]}" --priority-scheduling-preemption-threshold 1
row sj-mlfq-cache-aware-keepkv-gap2 "${MLFQ[@]}" --priority-scheduling-preemption-threshold 2
row sj-mlfq-cache-aware-keepkv-q40 "${MLFQ[@]}" --priority-scheduling-preemption-threshold 0 --sj-mlfq-base-quantum-ms 40
row sj-mlfq-cache-aware-keepkv-run2 "${MLFQ[@]}" --priority-scheduling-preemption-threshold 0
row fcfs-run2 --schedule-policy fcfs
date -u > "$marker"
