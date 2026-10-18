#!/usr/bin/env bash
# Point an SSH alias (ALIAS, default "runpod-sglang"; pod B is "runpod-sglang-b") at a
# new pod address (from get-pod's ssh.direct).
# Usage: [ALIAS=runpod-sglang-b] set_pod_ssh.sh <ip> <port> [pod-id]
set -euo pipefail
ip=$1 port=$2 pod=${3:-} alias=${ALIAS:-runpod-sglang}
python3 - "$ip" "$port" "$pod" "$alias" <<'EOF'
import pathlib, re, sys
ip, port, pod, alias = sys.argv[1:]
cfg = pathlib.Path.home() / ".ssh" / "config"
text = cfg.read_text()
block = re.search(rf"(?ms)^Host {alias}\n.*?(?=^\S|\Z)", text)
if block is None:
    sys.exit(f"no 'Host {alias}' block in ~/.ssh/config")
new = re.sub(r"(?m)^(\s+HostName ).*$", rf"\g<1>{ip}", block.group(0))
new = re.sub(r"(?m)^(\s+Port ).*$", rf"\g<1>{port}", new)
text = text[: block.start()] + new + text[block.end():]
cfg.write_text(text)
print(f"{alias} -> root@{ip}:{port}" + (f" (pod {pod})" if pod else ""))
EOF
