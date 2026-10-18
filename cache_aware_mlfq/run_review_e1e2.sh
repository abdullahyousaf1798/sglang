#!/usr/bin/env bash
# E1 (natural pool) and E2 (pinned pool) rows from REVIEW.md on one pod, back to back.
#   bash cache_aware_mlfq/run_review_e1e2.sh <e1-window-minute> <e2-window-minute>
# Writes /workspace/results/.review_e1e2_done when every row has finished.
set -uo pipefail
e1=$1 e2=$2
source /workspace/env.sh
cd /opt/sglang
marker=/workspace/results/.review_e1e2_done
rm -f "$marker"
CONFIGS="fcfs-mixed8k fcfs-mixed2k lpm-mixed2k spf-mixed2k fcfs-interval8 ours-gate-mixed2k ours-mixed2k" \
  bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$e1"
POOL=160000 CONFIGS="ours-gate-mixed2k-nopreempt ours-gate-mixed2k-thr3 ours-gate-mixed2k lpm-mixed2k spf-mixed2k" \
  bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 "$e2"
date -u > "$marker"
