# Pod-side environment: source it first in every pod command.
source /opt/venvs/sglang/bin/activate
export HF_HOME=/workspace/hf
export CARGO_HOME=/opt/build/cargo RUSTUP_HOME=/opt/build/rustup
export PATH="$CARGO_HOME/bin:$PATH"
cd /opt/sglang
