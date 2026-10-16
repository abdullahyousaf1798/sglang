#!/usr/bin/env bash
# Point the "runpod-sglang" SSH alias at a new pod address (from get-pod's ssh.direct).
# Usage: set_pod_ssh.sh <ip> <port> [pod-id]
set -euo pipefail
ip=$1 port=$2 pod=${3:-}
python3 - "$ip" "$port" "$pod" <<'EOF'
import pathlib, re, sys
ip, port, pod = sys.argv[1:]
cfg = pathlib.Path.home() / ".ssh" / "config"
text = cfg.read_text()
block = re.search(r"(?ms)^Host runpod-sglang\n.*?(?=^\S|\Z)", text)
if block is None:
    sys.exit("no 'Host runpod-sglang' block in ~/.ssh/config")
new = re.sub(r"(?m)^(\s+HostName ).*$", rf"\g<1>{ip}", block.group(0))
new = re.sub(r"(?m)^(\s+Port ).*$", rf"\g<1>{port}", new)
text = text[: block.start()] + new + text[block.end():]
if pod:
    text = re.sub(r"\(pod \w+\)", f"(pod {pod})", text)
cfg.write_text(text)
print(f"runpod-sglang -> root@{ip}:{port}")
EOF
