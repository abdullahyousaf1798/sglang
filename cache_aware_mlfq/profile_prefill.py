"""Measure prefill latency for a grid of (uncached tokens, cached prefix tokens).

Skip-join placement needs the time a request's first iteration takes, which
depends on both how many tokens must be prefilled and how long the cached prefix
they attend to is. For each grid cell this caches a random prefix, then times
requests that add a fresh random suffix and generate one token. Runs on the pod
with the same server setup as the benchmark rows, and writes the table to
cache_aware_mlfq/prefill_profile_<gpu>.json.
"""

import json
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_experiment import PORT, SERVER_ARGS, wait_for_server

UNCACHED = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CACHED = [0, 1024, 2048, 4096, 8192, 16384]
MAX_TOTAL = 24000  # stays under the 24576-token GPU pool
REPEATS = 3
# Ordinary Qwen3 vocabulary ids; special tokens start at 151643.
TOKEN_RANGE = (1000, 150000)


def generate(input_ids: list) -> dict:
    body = json.dumps({
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def flush_cache() -> None:
    # The server refuses (400) while the last request is still being released.
    for _ in range(20):
        req = urllib.request.Request(f"http://localhost:{PORT}/flush_cache", method="POST")
        try:
            urllib.request.urlopen(req, timeout=60).read()
            return
        except urllib.error.HTTPError as err:
            if err.code != 400:
                raise
            time.sleep(0.5)
    sys.exit("flush_cache kept failing: server reports pending requests")


def random_tokens(n: int) -> list:
    return [random.randrange(*TOKEN_RANGE) for _ in range(n)]


def timed_ms(input_ids: list) -> tuple:
    start = time.perf_counter()
    out = generate(input_ids)
    return (time.perf_counter() - start) * 1000, out["meta_info"].get("cached_tokens", 0)


def main() -> None:
    random.seed(0)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    log = open("/workspace/logs/profile_prefill_server.log", "w")
    server = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", *SERVER_ARGS],
                              stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(server, timeout_s=900)
        for _ in range(5):  # warm up kernels and CUDA graphs
            generate(random_tokens(512))
        overhead_ms = statistics.median(timed_ms(random_tokens(1))[0] for _ in range(10))

        cells = []
        for cached in CACHED:
            for uncached in UNCACHED:
                if cached + uncached > MAX_TOTAL:
                    continue
                flush_cache()
                prefix = random_tokens(cached)
                if cached:
                    generate(prefix)  # puts the prefix in the radix cache
                runs = [timed_ms(prefix + random_tokens(uncached)) for _ in range(REPEATS)]
                latency = statistics.median(ms for ms, _ in runs)
                hit = min(hit for _, hit in runs)
                cells.append({"uncached": uncached, "cached": cached,
                              "latency_ms": round(latency, 2), "cached_tokens_seen": hit})
                print(f"cached={cached:6d} uncached={uncached:5d} -> {latency:8.1f} ms "
                      f"(server saw {hit} cached)", flush=True)

        table = {"gpu": gpu, "model": SERVER_ARGS[SERVER_ARGS.index("--model-path") + 1],
                 "request_overhead_ms": round(overhead_ms, 2), "cells": cells}
        out = Path(__file__).resolve().parent / f"prefill_profile_{gpu.split()[-1].lower()}.json"
        out.write_text(json.dumps(table, indent=2))
        Path("/workspace/results").mkdir(exist_ok=True)
        (Path("/workspace/results") / out.name).write_text(out.read_text())
        print(f"wrote {out}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
