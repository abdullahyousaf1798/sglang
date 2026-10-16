#!/usr/bin/env bash
# Run after every pod start: restore the runtime snapshot onto local disk
# (a stop or a new pod starts with an empty container disk) and arm the idle stop.
set -euo pipefail
mkdir -p /workspace/logs
command -v zstd >/dev/null && command -v rsync >/dev/null \
  || (apt-get update -qq && apt-get install -y -qq zstd rsync)

if [ ! -x /opt/venvs/sglang/bin/python ]; then
  start=$(date +%s)
  zstd -dc /workspace/snapshots/runtime.tar.zst | tar -C / -xf -
  echo "restored runtime in $(( $(date +%s) - start )) s"
fi

pgrep -f "^bash /workspace/scripts/idle_stop.sh" >/dev/null \
  || (nohup setsid bash /workspace/scripts/idle_stop.sh >> /workspace/logs/idle_stop.log 2>&1 < /dev/null &)
echo READY
