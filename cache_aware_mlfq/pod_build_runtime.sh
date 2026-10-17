#!/usr/bin/env bash
# One-time: build the SGLang runtime on the pod's local disk and save it as one
# snapshot file on the network volume. Importing SGLang from the network volume
# takes ~5 minutes (slow per-file reads); from local disk it takes seconds, and
# the volume reads one big file at ~480 MB/s, so later pods restore the snapshot.
set -euxo pipefail
SNAP=/workspace/snapshots/runtime.tar.zst
export CARGO_HOME=/opt/build/cargo RUSTUP_HOME=/opt/build/rustup UV_CACHE_DIR=/opt/build/uv-cache
export HF_HOME=/workspace/hf
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p /workspace/logs /workspace/snapshots

# libssl-dev: HiCache's native hash extension includes <openssl/sha.h>.
apt-get update -qq
apt-get install -y -qq libssl-dev rsync zstd

# Base commit of the Mac branch; the Mac copy is synced on top of this checkout.
[ -d /opt/sglang ] || git clone --filter=blob:none \
  https://github.com/abdullahyousaf1798/sglang.git /opt/sglang
cd /opt/sglang
git checkout -B cache-aware-sj-mlfq bc76a6b819

[ -d /opt/venvs/sglang ] || uv venv --seed --python 3.12 /opt/venvs/sglang
source /opt/venvs/sglang/bin/activate

bash scripts/ci/utils/install_rust_protoc.sh
source "$CARGO_HOME/env"
uv pip install -e python --index-strategy unsafe-best-match --prerelease allow
# benchmark/hicache/bench_serving.py imports a video-dataset loader that needs PyAV.
uv pip install av
python -c 'from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash; get_native_hash([1], None)'

hf download Qwen/Qwen3-8B

# One server start fills the kernel caches under /root/.cache, so the snapshot carries them.
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 \
  > /workspace/logs/build_smoke_server.log 2>&1 &
server_pid=$!
for _ in $(seq 1 180); do
  curl -sf localhost:30000/health >/dev/null && break
  sleep 5
done
curl -sf localhost:30000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 8, "temperature": 0}}'
echo
kill "$server_pid"
wait "$server_pid" || true

# Rust build intermediates are not needed at runtime; the built .so files live in python/.
tar -C / --exclude='opt/sglang/rust/target' --exclude='opt/sglang/rust/*/target' \
  -cf - opt/sglang opt/venvs root/.cache | zstd -T0 -3 > "$SNAP.tmp"
mv "$SNAP.tmp" "$SNAP"
ls -la "$SNAP"
echo BUILD_DONE
