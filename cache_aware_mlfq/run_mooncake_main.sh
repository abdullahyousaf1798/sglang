#!/usr/bin/env bash
# Main table on Mooncake production traces (A100 80 GB): each configuration on each
# given trace window, window-major, so each window's rows run back to back on one pod.
# Runs on the pod in the background; writes a marker when every row has finished.
#
#   [POOL=<gpu-kv-tokens>] [CONFIGS="<config> ..."] bash cache_aware_mlfq/run_mooncake_main.sh \
#       <trace: toolagent|conversation> <slowdown> <window-minutes> <window-start-minute>...
#
# POOL pins the GPU KV pool (--max-total-tokens) and names the rows mooncake<POOL/1000>k-*;
# without it the pool is whatever fits, and rows are named mooncake-*.
set -uo pipefail
source /workspace/env.sh
trace=$1 slowdown=$2 minutes=$3
shift 3
marker=/workspace/results/.mooncake_main_done
rm -f "$marker"
PROFILE=/opt/sglang/cache_aware_mlfq/prefill_profile_a100-sxm4-80gb.json
MLFQ="--schedule-policy sj-mlfq --sj-mlfq-profile $PROFILE --enable-priority-scheduling
  --priority-scheduling-preemption-threshold 0 --retraction-policy priority"
# FastServe's mechanisms on SGLang: preempted KV is kept, evicted lowest level first
# after finished conversations (4b), and loaded back ahead of admission (5).
MECHANISMS="--preempt-keep-kv --radix-eviction-policy mlfq --sj-mlfq-load-ahead"
GATE=--sj-mlfq-prefill-gate
# The 2x2 behind the two contributions on top of FastServe: placement (prompt =
# FastServe's, cache-aware) x eviction (SGLang's LRU, two-tier "mlfq"), all with
# FastServe's own mechanisms ported (kept KV, load-ahead, and optionally the gate).
#   fastserve-lru        prompt       LRU       (FastServe ported)
#   ours-lru-eviction    cache-aware  LRU
#   fastserve            prompt       two-tier
#   ours                 cache-aware  two-tier
# and the same four with -gate (fastserve-lru-gate, ours-lru-gate, fastserve-gate, ours-gate).
PORTED="--preempt-keep-kv --sj-mlfq-load-ahead"
flags() {
  case $1 in
    fcfs) echo "--schedule-policy fcfs" ;;
    lpm) echo "--schedule-policy lpm" ;;
    shortest-prefill-first) echo "--schedule-policy shortest-prefill-first" ;;
    sj-mlfq-naive) echo "$MLFQ --sj-mlfq-cost prompt" ;;
    fastserve-lru) echo "$MLFQ --sj-mlfq-cost prompt $PORTED" ;;
    fastserve-lru-gate) echo "$MLFQ --sj-mlfq-cost prompt $PORTED $GATE" ;;
    fastserve) echo "$MLFQ --sj-mlfq-cost prompt $MECHANISMS" ;;
    fastserve-gate) echo "$MLFQ --sj-mlfq-cost prompt $MECHANISMS $GATE" ;;
    ours-lru-eviction) echo "$MLFQ --sj-mlfq-cost cache-aware $PORTED" ;;
    ours-lru-gate) echo "$MLFQ --sj-mlfq-cost cache-aware $PORTED $GATE" ;;
    ours) echo "$MLFQ --sj-mlfq-cost cache-aware $MECHANISMS" ;;
    ours-gate) echo "$MLFQ --sj-mlfq-cost cache-aware $MECHANISMS $GATE" ;;
    ours-no-load-ahead) echo "$MLFQ --sj-mlfq-cost cache-aware --preempt-keep-kv --radix-eviction-policy mlfq" ;;
    *) echo "unknown config: $1" >&2; return 1 ;;
  esac
}
DEFAULT_CONFIGS="fcfs lpm shortest-prefill-first sj-mlfq-naive fastserve ours ours-no-load-ahead ours-lru-eviction"
prefix=mooncake pool_flags=""
if [ -n "${POOL:-}" ]; then
  prefix="mooncake$((POOL / 1000))k" pool_flags="--max-total-tokens $POOL"
fi
for start in "$@"; do
  for config in ${CONFIGS:-$DEFAULT_CONFIGS}; do
    server_flags=$(flags "$config") || continue
    # Word splitting of the flag strings is intended; no flag contains a space.
    python cache_aware_mlfq/run_experiment.py --name "$prefix-$trace-$config-s$start" \
      --workload mooncake --trace "/workspace/data/mooncake_$trace.jsonl" \
      --start-min "$start" --minutes "$minutes" --slowdown "$slowdown" \
      -- $server_flags $pool_flags \
      || echo "FAILED: $config window $start"
  done
done
date -u > "$marker"
