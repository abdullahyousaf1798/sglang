#!/usr/bin/env bash
# Phase 2 of REVIEW.md (E3 and E4) on one pod. Waits for the E1/E2 rows to finish and
# for the Mac to signal that the final-design code has been synced, then runs the
# queue unattended. Pod a holds windows 10 and 40, pod b windows 25 and 55.
#   nohup bash cache_aware_mlfq/run_review_phase2.sh a|b &
set -uo pipefail
pod=$1
source /workspace/env.sh
cd /opt/sglang
R=/workspace/results
until [ -f $R/.review_e1e2_done ]; do sleep 60; done
# Code go-ahead; after two hours proceed anyway so a lost signal does not idle the pod.
for _ in $(seq 1 120); do [ -f $R/.phase2_go ] && break; sleep 60; done
echo "phase 2 start: $(date -u)"
rm -f /workspace/keep_running  # the idle stop may pause the pod once the queue drains
MAIN="fcfs lpm-mixed2k spf-mixed2k fcfs-mixed2k ours-final-nopreempt ours-final-thr3 fastserve-final-thr3"
NATURAL="fcfs lpm-mixed2k spf-mixed2k fcfs-mixed2k ours-final-thr3 fastserve-final-thr3"
KEY="fcfs spf-mixed2k ours-final-thr3"
if [ "$pod" = a ]; then
  # E3: chunk yield on and off, 0.4-minute window 10, both pools
  CONFIGS="ours-final-thr3 ours-final-thr3-noyield" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 10
  POOL=160000 CONFIGS="ours-final-thr3 ours-final-thr3-noyield" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 0.4 10
  # E4: 1-minute windows
  POOL=160000 TAG=m1 CONFIGS="$MAIN" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 10 40
  TAG=m1 CONFIGS="$NATURAL" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 10
  POOL=160000 TAG=m1-r2 CONFIGS="$KEY" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 10 40
  TAG=m1 CONFIGS="$KEY" bash cache_aware_mlfq/run_mooncake_main.sh conversation 11 1.0 10
else
  POOL=160000 TAG=m1 CONFIGS="$MAIN" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 25 55
  TAG=m1 CONFIGS="$NATURAL" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 25
  POOL=160000 TAG=m1-r2 CONFIGS="$KEY" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 11 1.0 25 55
  # load sweep on the heaviest window
  POOL=160000 TAG=m1-sd15 CONFIGS="$KEY" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 15 1.0 55
  POOL=160000 TAG=m1-sd9 CONFIGS="$KEY" bash cache_aware_mlfq/run_mooncake_main.sh toolagent 9 1.0 55
fi
date -u > $R/.review_phase2_done
echo "phase 2 done: $(date -u)"
