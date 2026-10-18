#!/usr/bin/env bash
# Remaining review phases on one pod, after the E1 rows already running finish:
# E1b (8k mixed baselines), E2 (churn settings), E3 (chunk yield), E4 (1-minute windows).
# Pod a holds windows 10 and 40, pod b windows 25 and 55.
#   nohup bash cache_aware_mlfq/run_review_overnight.sh a|b &
set -uo pipefail
pod=$1
source /workspace/env.sh
cd /opt/sglang
R=/workspace/results
# The running E1 invocation of run_mooncake_main.sh writes this when its last row is done.
until [ -f $R/.mooncake_main_done ]; do sleep 30; done
echo "overnight queue start: $(date -u)"
rm -f /workspace/keep_running  # the idle stop may pause the pod once the queue drains
RUN="bash cache_aware_mlfq/run_mooncake_main.sh"
if [ "$pod" = a ]; then e1=10 e2=10 w="10 40" ; else e1=25 e2=55 w="25 55"; fi
# E1b: natural pool, 0.4-minute window
CONFIGS="lpm-mixed8k spf-mixed8k ours-gate-mixed8k" $RUN toolagent 11 0.4 $e1
# E2: pinned pool, three preemption settings against the native rows
POOL=160000 CONFIGS="ours-gate-mixed8k-nopreempt ours-gate-mixed8k-thr3 ours-gate-mixed8k lpm-mixed8k spf-mixed8k" $RUN toolagent 11 0.4 $e2
date -u > $R/.review_e1e2_done
MAIN="fcfs lpm-mixed8k spf-mixed8k fcfs-mixed8k ours-final-nopreempt ours-final-thr3 fastserve-final-thr3"
NATURAL="fcfs lpm-mixed8k spf-mixed8k fcfs-mixed8k ours-final-thr3 fastserve-final-thr3"
KEY="fcfs spf-mixed8k ours-final-thr3"
if [ "$pod" = a ]; then
  # E3: chunk yield on and off, 0.4-minute window 10, both pools
  CONFIGS="ours-final-thr3 ours-final-thr3-noyield" $RUN toolagent 11 0.4 10
  POOL=160000 CONFIGS="ours-final-thr3 ours-final-thr3-noyield" $RUN toolagent 11 0.4 10
fi
# E4: 1-minute windows
POOL=160000 TAG=m1 CONFIGS="$MAIN" $RUN toolagent 11 1.0 $w
TAG=m1 CONFIGS="$NATURAL" $RUN toolagent 11 1.0 $e1
POOL=160000 TAG=m1-r2 CONFIGS="$KEY" $RUN toolagent 11 1.0 $w
if [ "$pod" = a ]; then
  TAG=m1 CONFIGS="$KEY" $RUN conversation 11 1.0 10
else
  # load sweep on the heaviest window
  POOL=160000 TAG=m1-sd15 CONFIGS="$KEY" $RUN toolagent 15 1.0 55
  POOL=160000 TAG=m1-sd9 CONFIGS="$KEY" $RUN toolagent 9 1.0 55
fi
date -u > $R/.review_phase2_done
echo "overnight queue done: $(date -u)"
