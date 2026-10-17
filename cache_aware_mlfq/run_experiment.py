"""Run one benchmark configuration on the pod and save its results.

Starts an SGLang server with the shared server setup plus this run's scheduling
flags, waits until it is ready, runs a multi-turn chat workload against it,
then records the client's summary and the server's cache counters under
/workspace/results/<name>/ and stops the server.

Workloads:
  sharegpt   real multi-turn ShareGPT conversations replayed through the chat API
             (benchmark/hicache/bench_serving.py); the main workload.
  synthetic  random-token conversations with uniform lengths
             (benchmark/hicache/bench_multiturn.py); the earlier rows.

    python cache_aware_mlfq/run_experiment.py --name fcfs-sharegpt-s1 --workload sharegpt \
        --seed 1 -- --schedule-policy fcfs
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

# Mooncake rows run on an A100 80 GB with the KV pool at whatever fits, as deployed.
# Trace prompts reach 126k tokens, past Qwen3-8B's native 32k (40k positions), so
# the server uses Qwen3's documented YaRN extension to 131k. SGLang derives the
# context length from max_position_embeddings alone when the YaRN config names the
# original length, so that is raised to match.
MOONCAKE_SERVER_ARGS = [
    "--model-path", MODEL,
    "--port", str(PORT),
    "--mem-fraction-static", "0.88",
    "--context-length", "131072",
    "--json-model-override-args", json.dumps({
        "max_position_embeddings": 131072,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                         "original_max_position_embeddings": 32768}}),
    "--enable-hierarchical-cache",
    "--hicache-size", "80",  # fits both pods (125 and 251 GB of host RAM)
    "--enable-metrics",
]


def server_args(workload_name: str) -> list:
    return MOONCAKE_SERVER_ARGS if workload_name == "mooncake" else SERVER_ARGS


# A window of a Mooncake trace (mooncake_replay.py), replayed at its own arrival
# times stretched by "slowdown"; every request's prompt and output length are real.
MOONCAKE_WORKLOAD = {
    "trace": "/workspace/data/mooncake_toolagent.jsonl",
    "start-min": 0.0,
    "minutes": 2.0,
    "slowdown": 4.0,
}

# 64 conversations x 6 turns. Each turn adds 256-1024 new tokens and gets a
# 64-256 token answer, so contexts grow to several thousand tokens and together
# far exceed the 24k-token GPU pool.
SYNTHETIC_WORKLOAD = {
    "num-clients": 64,
    "num-rounds": 6,
    "request-length": 1024,
    "output-length": 256,
    "range-ratio": 0.25,
    "max-parallel": 64,
    "request-rate": 2.0,
    "seed": 1,
}

# Real conversations whose whole history fits in 16k tokens (prepare_sharegpt.py),
# with each reply as long as the real one. A conversation's next turn is queued as
# soon as its previous answer arrives; requests are released as a Poisson process.
SHAREGPT_DATA = "/workspace/data/sharegpt_multiturn_16k.json"
SHAREGPT_WORKLOAD = {
    "num-prompts": 200,  # conversations
    "request-rate": 1.0,
    "seed": 1,
}
# Qwen3 thinks by default; ordinary chat replies keep reply lengths meaningful.
CHAT_REQUEST_BODY = '{"chat_template_kwargs": {"enable_thinking": false}}'


def bench_command(workload_name: str, workload: dict, out: Path, tag: str) -> list:
    if workload_name == "mooncake":
        cmd = [
            sys.executable, "cache_aware_mlfq/mooncake_replay.py", "--port", str(PORT),
            "--output-file", str(out / "bench.jsonl"),
            "--requests-file", str(out / "requests.jsonl"),
        ]
    elif workload_name == "sharegpt":
        cmd = [
            sys.executable, "benchmark/hicache/bench_serving.py",
            "--backend", "sglang", "--port", str(PORT), "--model", MODEL,
            "--dataset-name", "sharegpt", "--dataset-path", SHAREGPT_DATA,
            "--enable-multiturn", "--extra-request-body", CHAT_REQUEST_BODY,
            "--output-file", str(out / "bench.jsonl"), "--disable-tqdm",
        ]
    else:
        cmd = [
            sys.executable, "benchmark/hicache/bench_multiturn.py",
            "--model-path", MODEL, "--port", str(PORT),
            "--distribution", "poisson", "--disable-auto-run",
            "--log-file", str(out / "bench.jsonl"), "--tag", tag,
        ]
    for key, value in workload.items():
        cmd += [f"--{key}", str(value)]
    return cmd


def normalize_summary(workload_name: str, raw: dict) -> dict:
    """Every client's results in one shape: seconds, under "summary", as make_table reads."""
    if workload_name != "sharegpt":
        return raw  # bench_multiturn and mooncake_replay already write this shape
    return {
        "summary": {
            "total_requests": raw["completed"],
            "average_latency": raw["mean_e2e_latency_ms"] / 1000,
            "p99_latency": raw["p99_e2e_latency_ms"] / 1000,
            "average_ttft": raw["mean_ttft_ms"] / 1000,
            "p99_ttft": raw["p99_ttft_ms"] / 1000,
            "average_tpot": raw["mean_tpot_ms"] / 1000,
            "throughput": raw["request_throughput"],
        },
        "raw": raw,
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
    # Requests whose KV was released and requeued (decode retraction or priority
    # preemption), and the decoded tokens they lost and must recompute.
    for metric, key in [
        ("sglang:num_retracted_requests_total", "retracted_requests"),
        ("sglang:num_retracted_output_tokens_total", "retracted_output_tokens"),
    ]:
        values = re.findall(rf"{metric}\{{[^}}]*\}} ([0-9.e+]+)", text)
        counters[key] = sum(float(v) for v in values)
    # All prefill compute, including requests prefilled again after preemption or
    # retraction; prefill_tokens_input above counts first attempts only.
    values = re.findall(
        r'sglang:realtime_tokens_total\{[^}]*mode="prefill_compute"[^}]*\} ([0-9.e+]+)', text
    )
    counters["prefill_compute_tokens_all"] = sum(float(v) for v in values)
    # Host-to-device (CPU memory to GPU) load-back of cached prefixes: tokens copied,
    # copies, and their GPU-stream time; the latency proactive load-back would hide.
    for metric, key in [
        (r'sglang:load_back_tokens_total\{[^}]*pool="kv"[^}]*\}', "load_back_tokens"),
        (r"sglang:load_back_duration_seconds_count\{[^}]*\}", "load_back_copies"),
        (r"sglang:load_back_duration_seconds_sum\{[^}]*\}", "load_back_seconds"),
    ]:
        values = re.findall(rf"{metric} ([0-9.e+]+)", text)
        counters[key] = sum(float(v) for v in values)
    return counters


def read_load_ahead(server_log: Path) -> dict:
    """Last cumulative load-ahead report in the server log (logged after every copy)."""
    reports = re.findall(r"sj-mlfq load-ahead: (\d+) copies, (\d+) tokens", server_log.read_text())
    if not reports:
        return {}
    copies, tokens = reports[-1]
    return {"load_ahead_copies": int(copies), "load_ahead_tokens": int(tokens)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results folder name, e.g. fcfs")
    parser.add_argument("--workload", choices=["synthetic", "sharegpt", "mooncake"],
                        default="synthetic")
    parser.add_argument("--request-rate", type=float, help="requests per second")
    parser.add_argument("--seed", type=int, help="client seed; vary it across repeats")
    parser.add_argument("--num-conversations", type=int, help="sharegpt only")
    parser.add_argument("--trace", help="mooncake only: trace file")
    parser.add_argument("--start-min", type=float, help="mooncake only: window start (trace minutes)")
    parser.add_argument("--minutes", type=float, help="mooncake only: window length (trace minutes)")
    parser.add_argument("--slowdown", type=float, help="mooncake only: arrival-time stretch")
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

    workload = dict({"sharegpt": SHAREGPT_WORKLOAD, "mooncake": MOONCAKE_WORKLOAD}.get(
        args.workload, SYNTHETIC_WORKLOAD))
    overrides = {
        "request-rate": args.request_rate,
        "seed": args.seed,
        "num-prompts": args.num_conversations,
        "trace": args.trace,
        "start-min": args.start_min,
        "minutes": args.minutes,
        "slowdown": args.slowdown,
    }
    for key, value in overrides.items():
        if value is not None:
            if key not in workload:
                sys.exit(f"--{key} does not apply to the {args.workload} workload")
            workload[key] = value
    config = {
        "name": args.name,
        "workload_name": args.workload,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "server_args": server_args(args.workload) + server_flags,
        "workload": workload,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    with open(out / "server.log", "w") as server_log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", *server_args(args.workload), *server_flags],
            stdout=server_log, stderr=subprocess.STDOUT, cwd=REPO,
        )
        try:
            wait_for_server(server, timeout_s=900)
            bench_cmd = bench_command(args.workload, workload, out, args.name)
            started = time.time()
            with open(out / "bench.log", "w") as bench_log:
                bench = subprocess.Popen(bench_cmd, stdout=bench_log, stderr=subprocess.STDOUT,
                                         cwd=REPO)
                # The client waits forever on a dead server, which would keep the pod busy.
                while bench.poll() is None:
                    if server.poll() is not None:
                        bench.kill()
                        sys.exit(f"server exited with code {server.returncode} during the benchmark")
                    time.sleep(5)
                if bench.returncode != 0:
                    sys.exit(f"benchmark client failed with code {bench.returncode}")
            time.sleep(10)  # let the server's periodic metrics logging catch up
            raw = json.loads((out / "bench.jsonl").read_text().splitlines()[-1])
            summary = normalize_summary(args.workload, raw)
            summary["wall_time_s"] = time.time() - started
            summary["server_counters"] = read_cache_counters() | read_load_ahead(out / "server.log")
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
