#!/usr/bin/env bash
# Step 2: SGLang's existing policies on the shared workload, then the prefill
# latency profile that step 3's placement uses. Runs on the pod in the
# background; writes a marker file when everything has finished.
set -uo pipefail
source /workspace/env.sh
marker=/workspace/results/.baselines_done
rm -f "$marker"
for policy in "$@"; do
  python cache_aware_mlfq/run_experiment.py --name "$policy" -- --schedule-policy "$policy" \
    || echo "FAILED: $policy"
done
python cache_aware_mlfq/profile_prefill.py || echo "FAILED: profile_prefill"
date -u > "$marker"
