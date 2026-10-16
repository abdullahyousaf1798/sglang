"""Run one benchmark configuration on the pod and save its results.

Starts an SGLang server with the shared server setup plus this run's scheduling
flags, waits until it is ready, runs SGLang's multi-turn chat client against it,
then records the client's summary and the server's cache counters under
/workspace/results/<name>/ and stops the server.

    python cache_aware_mlfq/run_experiment.py --name fcfs -- --schedule-policy fcfs
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path("/workspace/results")
MODEL = "Qwen/Qwen3-8B"
PORT = 30000

# Shared by every row so rows differ only in their scheduling flags. The GPU KV
# pool is pinned in tokens (not a memory fraction) so it is the same on any GPU,
# and kept small so idle conversations spill to the CPU-memory tier (HiCache).
SERVER_ARGS = [
    "--model-path", MODEL,
    "--port", str(PORT),
    "--mem-fraction-static", "0.88",
    "--max-total-tokens", "24576",
    "--enable-hierarchical-cache",
    "--hicache-size", "30",
    "--enable-metrics",
]

# 64 conversations x 6 turns. Each turn adds 256-1024 new tokens and gets a
# 64-256 token answer, so contexts grow to several thousand tokens and together
# far exceed the 24k-token GPU pool.
WORKLOAD = {
    "num-clients": 64,
    "num-rounds": 6,
    "request-length": 1024,
    "output-length": 256,
    "range-ratio": 0.25,
    "max-parallel": 64,
    "request-rate": 2.0,
    "seed": 1,
}


def wait_for_server(proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited with code {proc.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5):
                return
        except OSError:
            time.sleep(5)
    sys.exit(f"server not ready after {timeout_s:.0f} s")


def read_cache_counters() -> dict:
    """Prefill token totals by source (input/device_hit/host_hit) and retractions."""
    with urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    counters = {}
    for mode in re.findall(r'sglang:prefill_effective_tokens_total\{[^}]*mode="(\w+)"', text):
        values = re.findall(
            rf'sglang:prefill_effective_tokens_total\{{[^}}]*mode="{mode}"[^}}]*\}} ([0-9.e+]+)', text
        )
        counters[f"prefill_tokens_{mode}"] = sum(float(v) for v in values)
    retracted = re.findall(r"sglang:num_retracted_requests_total\{[^}]*\} ([0-9.e+]+)", text)
    counters["retracted_requests"] = sum(float(v) for v in retracted)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results folder name, e.g. fcfs")
    parser.add_argument("--request-rate", type=float, default=WORKLOAD["request-rate"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("server_flags", nargs=argparse.REMAINDER,
                        help="scheduling flags after --, e.g. -- --schedule-policy lpm")
    args = parser.parse_args()
    server_flags = [f for f in args.server_flags if f != "--"]

    out = RESULTS / args.name
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.jsonl").unlink(missing_ok=True)

    workload = {**WORKLOAD, "request-rate": args.request_rate}
    config = {
        "name": args.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "server_args": SERVER_ARGS + server_flags,
        "workload": workload,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    with open(out / "server.log", "w") as server_log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", *SERVER_ARGS, *server_flags],
            stdout=server_log, stderr=subprocess.STDOUT, cwd=REPO,
        )
        try:
            wait_for_server(server, timeout_s=900)
            bench_cmd = [
                sys.executable, "benchmark/hicache/bench_multiturn.py",
                "--model-path", MODEL, "--port", str(PORT),
                "--distribution", "poisson", "--disable-auto-run",
                "--log-file", str(out / "bench.jsonl"), "--tag", args.name,
            ]
            for key, value in workload.items():
                bench_cmd += [f"--{key}", str(value)]
            started = time.time()
            with open(out / "bench.log", "w") as bench_log:
                subprocess.run(bench_cmd, stdout=bench_log, stderr=subprocess.STDOUT,
                               cwd=REPO, check=True)
            time.sleep(10)  # let the server's periodic metrics logging catch up
            summary = json.loads((out / "bench.jsonl").read_text().splitlines()[-1])
            summary["wall_time_s"] = time.time() - started
            summary["server_counters"] = read_cache_counters()
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"done: {out}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
