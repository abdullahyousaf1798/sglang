#!/usr/bin/env bash
# Mac side: prepare the shared volume from the build pod and start the runtime build.
#   bash cache_aware_mlfq/pod_bootstrap.sh <ssh-alias>
set -euo pipefail
host=$1
cd "$(dirname "$0")"
ssh "$host" 'mkdir -p /workspace/scripts /workspace/logs /workspace/results /workspace/data /workspace/snapshots; rm -f /workspace/snapshots/BUILD_DONE'
scp -q pod_build_runtime.sh pod_resume.sh idle_stop.sh pod_wait_and_launch.sh "$host:/workspace/scripts/"
scp -q pod_env.sh "$host:/workspace/env.sh"
ssh "$host" 'cd /workspace/data && for t in toolagent conversation; do [ -s mooncake_$t.jsonl ] || curl -sSL -o mooncake_$t.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/${t}_trace.jsonl; done; wc -l mooncake_*.jsonl'
ssh "$host" 'nohup setsid bash /workspace/scripts/pod_build_runtime.sh > /workspace/logs/build.log 2>&1 < /dev/null & echo build started'
