#!/usr/bin/env bash
# Copy our changes from the Mac working tree (the source of truth) onto the pod's
# checkout of the base commit. Only files that differ from the base are sent:
# a full rsync compares every SGLang file over the long link and takes ~8 minutes.
set -euo pipefail
HOST=${POD_HOST:-runpod-sglang}
BASE=bc76a6b819
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

# Provenance: every run records the commit its code came from (run_experiment.py).
stamp=cache_aware_mlfq/SYNC_COMMIT
printf '%s %s\n' "$(git rev-parse HEAD)" "$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo clean || echo dirty)" > "$stamp"
changed=$( { git diff --name-only --diff-filter=d "$BASE"; git ls-files --others --exclude-standard; echo "$stamp"; } \
  | grep -v '^cache_aware_mlfq/results/' | sort -u || true)
deleted=$(git diff --name-only --diff-filter=D "$BASE" || true)

ssh "$HOST" 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)'
if [ -n "$changed" ]; then
  printf '%s\n' "$changed" | rsync -az --files-from=- ./ "$HOST:/opt/sglang/"
fi

# Undo pod-side edits the Mac no longer has, and apply Mac-side deletions.
{ printf '%s\n' "$changed"; printf '%s\n' "$deleted" | sed 's/^/DELETE /'; } | ssh "$HOST" '
  cd /opt/sglang
  grep -v "^DELETE " | grep . | sort -u > /tmp/mac_changed || true
  { git diff --name-only; git ls-files --others --exclude-standard; } \
    | grep -v "^cache_aware_mlfq/results/" | sort -u > /tmp/pod_changed
  comm -13 /tmp/mac_changed /tmp/pod_changed | while read -r f; do
    if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then git checkout -q -- "$f"; else rm -f -- "$f"; fi
  done'
if [ -n "$deleted" ]; then
  printf '%s\n' "$deleted" | ssh "$HOST" 'cd /opt/sglang && xargs -r rm -f --'
fi
echo "synced $(printf '%s\n' "$changed" | grep -c . || true) changed, $(printf '%s\n' "$deleted" | grep -c . || true) deleted -> $HOST:/opt/sglang"
