#!/usr/bin/env bash
# Stop this pod once nothing has run for IDLE_LIMIT_MIN minutes (default 20),
# unless /workspace/keep_running (or keep_running.<pod id>, on a shared volume) exists. "Running" means an SGLang server, a
# benchmark, a profiling job, any process on the GPU, or an open SSH session
# (a long command run over SSH, such as a restore, is work in progress).
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=" | xargs)
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null
limit=${IDLE_LIMIT_MIN:-20}
idle=0
while true; do
  if [ -e /workspace/keep_running ] || [ -e "/workspace/keep_running.$RUNPOD_POD_ID" ] \
    || pgrep -f "sglang.launch_server|bench_multiturn|run_experiment|profile_prefill" >/dev/null \
    || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] \
    || pgrep -f "^sshd: root@" >/dev/null; then
    idle=0
  else
    idle=$((idle + 1))
  fi
  if [ "$idle" -ge "$limit" ]; then
    echo "$(date -u) idle for $limit min; stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID"
    exit 0
  fi
  sleep 60
done
