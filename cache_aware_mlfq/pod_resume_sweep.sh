#!/usr/bin/env bash
# One-off: after the sweep scripts were stopped, let the row in progress finish,
# then run the 160k-token pool 2x2 sweep on this pod's windows (rows already done
# are skipped: run_experiment refuses to overwrite them), then any natural-pool
# gate rows still owed.
#
#   [NATURAL_GATE_WINDOW=<minute>] bash cache_aware_mlfq/pod_resume_sweep.sh <pressured-window>...
set -uo pipefail
source /workspace/env.sh
while pgrep -f "[r]un_experiment.py" >/dev/null; do sleep 15; done
# Most important rows first; the experimental shortest-prefill-first crashed once
# under this pool, so it runs last.
POOL=160000 CONFIGS="fcfs lpm fastserve-lru fastserve ours-lru-eviction ours fastserve-lru-gate fastserve-gate ours-lru-gate ours-gate shortest-prefill-first" \
  bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$@"
if [ -n "${NATURAL_GATE_WINDOW:-}" ]; then
  CONFIGS="fastserve-gate ours-gate" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$NATURAL_GATE_WINDOW"
fi
