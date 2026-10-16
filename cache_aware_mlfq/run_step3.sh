#!/usr/bin/env bash
# Step 3: skip-join MLFQ with FastServe's full-prompt placement and with
# cache-aware placement. Runs on the pod in the background; writes a marker
# file when both rows have finished.
set -uo pipefail
source /workspace/env.sh
marker=/workspace/results/.step3_done
rm -f "$marker"
# A one-level gap is enough to preempt, and decode-time retraction drops the
# lowest level first, so both preemption paths follow MLFQ order.
MLFQ=(
  --schedule-policy sj-mlfq
  --sj-mlfq-profile /opt/sglang/cache_aware_mlfq/prefill_profile_4090.json
  --enable-priority-scheduling
  --priority-scheduling-preemption-threshold 0
  --retraction-policy priority
)
for cost in prompt cache-aware; do
  python cache_aware_mlfq/run_experiment.py --name "sj-mlfq-$cost" -- "${MLFQ[@]}" --sj-mlfq-cost "$cost" \
    || echo "FAILED: sj-mlfq-$cost"
done
date -u > "$marker"
