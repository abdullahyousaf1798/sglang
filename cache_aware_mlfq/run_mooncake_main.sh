#!/usr/bin/env bash
# Main table on Mooncake production traces (A100 80 GB): each configuration on each
# given trace window, window-major, so each window's rows run back to back on one pod.
# Runs on the pod in the background; writes a marker when every row has finished.
#
#   [POOL=<gpu-kv-tokens>] [CONFIGS="<config> ..."] [TAG=<suffix>] bash cache_aware_mlfq/run_mooncake_main.sh \
#       <trace: toolagent|conversation> <slowdown> <window-minutes> <window-start-minute>...
#
# POOL pins the GPU KV pool (--max-total-tokens) and names the rows mooncake<POOL/1000>k-*;
# without it the pool is whatever fits, and rows are named mooncake-*. TAG is appended to
# the row name (e.g. m1 for 1-minute windows, m1-r2 for a repeat, m1-sd9 for slowdown 9).
set -uo pipefail
source /workspace/env.sh
trace=$1 slowdown=$2 minutes=$3
shift 3
marker=/workspace/results/.mooncake_main_done
rm -f "$marker"
PROFILE=/opt/sglang/cache_aware_mlfq/prefill_profile_a100-sxm4-80gb.json
MLFQ_BASE="--schedule-policy sj-mlfq --sj-mlfq-profile $PROFILE --enable-priority-scheduling
  --retraction-policy priority"
MLFQ="$MLFQ_BASE --priority-scheduling-preemption-threshold 0"
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
# Review rows (REVIEW.md, E1/E2): SGLang's own Sarathi-style batching and cadence knob
# as baselines, and the reduced design (cache-aware MLFQ + gate + kept KV, LRU eviction,
# no load-ahead) on top of mixed 2k chunks with three preemption settings.
MIXED2K="--enable-mixed-chunk --chunked-prefill-size 2048"
# E1 showed 2k chunks overload this trace (4x the steps, each paying the decode rows'
# attention); the default 8k chunk keeps throughput, so the later phases use it.
MIXED8K="--enable-mixed-chunk"
OURS="$MLFQ_BASE --sj-mlfq-cost cache-aware --preempt-keep-kv"
# E3/E4 "final" rows: the reduced design of REVIEW.md. No profile (the cost model is
# fitted online), kept KV, the gate, mixed 8k chunks and chunk yield; LRU eviction.
MLFQ_ONLINE="--schedule-policy sj-mlfq --enable-priority-scheduling --retraction-policy priority"
FINAL="$MLFQ_ONLINE --preempt-keep-kv $GATE $MIXED8K --sj-mlfq-chunk-yield"
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
    # E1: the bar
    fcfs-mixed8k) echo "--schedule-policy fcfs --enable-mixed-chunk" ;;
    fcfs-mixed2k) echo "--schedule-policy fcfs $MIXED2K" ;;
    lpm-mixed2k) echo "--schedule-policy lpm $MIXED2K" ;;
    spf-mixed2k) echo "--schedule-policy shortest-prefill-first $MIXED2K" ;;
    fcfs-interval8) echo "--schedule-policy fcfs --prefill-decode-interval 8" ;;
    ours-gate-mixed2k) echo "$OURS --priority-scheduling-preemption-threshold 0 $GATE $MIXED2K" ;;
    ours-mixed2k) echo "$OURS --priority-scheduling-preemption-threshold 0 $MIXED2K" ;;
    ours-gate-mixed2k-nopreempt) echo "$OURS --disable-priority-preemption $GATE $MIXED2K" ;;
    ours-gate-mixed2k-thr3) echo "$OURS --priority-scheduling-preemption-threshold 3 $GATE $MIXED2K" ;;
    # E1b: the SGLang-native candidates and the gate on 8k mixed chunks
    lpm-mixed8k) echo "--schedule-policy lpm $MIXED8K" ;;
    spf-mixed8k) echo "--schedule-policy shortest-prefill-first $MIXED8K" ;;
    ours-gate-mixed8k) echo "$OURS --priority-scheduling-preemption-threshold 0 $GATE $MIXED8K" ;;
    # E2: churn control (profile-based MLFQ, as in E1, so only preemption changes)
    ours-gate-mixed8k-nopreempt) echo "$OURS --disable-priority-preemption $GATE $MIXED8K" ;;
    ours-gate-mixed8k-thr3) echo "$OURS --priority-scheduling-preemption-threshold 3 $GATE $MIXED8K" ;;
    # E3/E4: final design, with FastServe placement as the ablation and no-yield for E3
    ours-final-nopreempt) echo "$FINAL --sj-mlfq-cost cache-aware --disable-priority-preemption" ;;
    ours-final-thr3) echo "$FINAL --sj-mlfq-cost cache-aware --priority-scheduling-preemption-threshold 3" ;;
    fastserve-final-thr3) echo "$FINAL --sj-mlfq-cost prompt --priority-scheduling-preemption-threshold 3" ;;
    ours-final-thr3-noyield) echo "$MLFQ_ONLINE --preempt-keep-kv $GATE $MIXED8K --sj-mlfq-cost cache-aware --priority-scheduling-preemption-threshold 3" ;;
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
    python cache_aware_mlfq/run_experiment.py --name "$prefix-$trace-$config-s$start${TAG:+-$TAG}" \
      --workload mooncake --trace "/workspace/data/mooncake_$trace.jsonl" \
      --start-min "$start" --minutes "$minutes" --slowdown "$slowdown" \
      -- $server_flags $pool_flags \
      || echo "FAILED: $config window $start"
  done
done
date -u > "$marker"
