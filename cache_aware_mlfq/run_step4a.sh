#!/usr/bin/env bash
# Step 4a: cache-aware skip-join MLFQ with preempted requests' KV kept in the
# prefix cache. Runs on the pod in the background; writes a marker when done.
set -uo pipefail
source /workspace/env.sh
marker=/workspace/results/.step4a_done
rm -f "$marker"
MLFQ=(
  --schedule-policy sj-mlfq
  --sj-mlfq-profile /opt/sglang/cache_aware_mlfq/prefill_profile_4090.json
  --sj-mlfq-cost cache-aware
  --enable-priority-scheduling
  --priority-scheduling-preemption-threshold 0
  --retraction-policy priority
)
python cache_aware_mlfq/run_experiment.py --name sj-mlfq-cache-aware-keepkv -- "${MLFQ[@]}" --preempt-keep-kv \
  || echo "FAILED: sj-mlfq-cache-aware-keepkv"
date -u > "$marker"
