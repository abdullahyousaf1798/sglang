"""Replay a window of a Mooncake trace against an SGLang server, keeping its prefix sharing.

Mooncake traces (Kimi production traffic, FAST'25) give each request an arrival
timestamp, input and output lengths, and the ids of the 512-token blocks its prompt
is made of; requests that share a block id share that part of the prompt. Here each
block id becomes a fixed block of token ids, so shared ids are real shared prefixes
(SGLang's built-in Mooncake replay shrinks each block to ~130 filler tokens). Requests
go to the native /generate API as token ids, at their trace times stretched by
--slowdown, and each generates exactly its trace output length.

    python cache_aware_mlfq/mooncake_replay.py --trace /workspace/data/mooncake_toolagent.jsonl \
        --start-min 0 --minutes 2 --slowdown 4 --output-file bench.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

BLOCK_TOKENS = 512
# Ordinary Qwen3 vocabulary ids; special tokens start at 151643.
TOKEN_RANGE = (1000, 150000)


class PromptBuilder:
    def __init__(self):
        self._blocks: dict = {}

    def _block(self, block_id: int) -> list:
        if block_id not in self._blocks:
            rng = random.Random(block_id)
            self._blocks[block_id] = [rng.randrange(*TOKEN_RANGE) for _ in range(BLOCK_TOKENS)]
        return self._blocks[block_id]

    def build(self, record: dict) -> list:
        ids = [t for block_id in record["hash_ids"] for t in self._block(block_id)]
        return ids[: record["input_length"]]


def load_window(path: str, start_min: float, minutes: float) -> list:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    begin, end = start_min * 60_000, (start_min + minutes) * 60_000
    window = sorted((r for r in records if begin <= r["timestamp"] < end),
                    key=lambda r: r["timestamp"])
    for r in window:
        r["offset_s"] = (r["timestamp"] - begin) / 1000
    return window


async def send(session: aiohttp.ClientSession, url: str, input_ids: list, output_len: int) -> dict:
    body = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": output_len, "ignore_eos": True, "temperature": 0.0},
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    completion_tokens = 0
    async with session.post(url, json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data:"):])
            completion_tokens = chunk["meta_info"]["completion_tokens"]
            if ttft is None and completion_tokens > 0:
                ttft = time.perf_counter() - start
    e2e = time.perf_counter() - start
    return {"e2e_s": e2e, "ttft_s": e2e if ttft is None else ttft,
            "completion_tokens": completion_tokens}


async def replay(args) -> list:
    import aiohttp  # here, so analyze_mooncake.py can reuse load_window without it

    window = load_window(args.trace, args.start_min, args.minutes)
    builder = PromptBuilder()
    prompts = [builder.build(r) for r in window]  # built up front so sending stays on time
    url = f"http://localhost:{args.port}/generate"
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    results = [None] * len(window)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(i: int) -> None:
            record = window[i]
            output_len = max(1, record["output_length"])
            await asyncio.sleep(max(0.0, t0 + record["offset_s"] * args.slowdown - time.perf_counter()))
            sent_at = time.perf_counter() - t0
            try:
                res = await send(session, url, prompts[i], output_len)
                res["error"] = None
            except Exception as err:  # recorded, so a failed request shows in the summary
                res = {"e2e_s": None, "ttft_s": None, "completion_tokens": 0, "error": repr(err)}
            results[i] = {"index": i, "sent_at_s": sent_at, "input_len": len(prompts[i]),
                          "output_len": output_len, **res}

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(len(window))))
        duration = time.perf_counter() - t0
    return results, duration


def percentile(values: list, p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def summarize(results: list, duration_s: float, args) -> dict:
    done = [r for r in results if r["error"] is None]
    e2e = [r["e2e_s"] for r in done]
    ttft = [r["ttft_s"] for r in done]
    tpot = [(r["e2e_s"] - r["ttft_s"]) / (r["output_len"] - 1) for r in done if r["output_len"] > 1]
    # FastServe's metric: end-to-end latency divided by output length.
    per_token = [r["e2e_s"] / r["output_len"] for r in done]
    return {
        "summary": {
            "total_requests": len(results),
            "completed": len(done),
            "failed": len(results) - len(done),
            "average_latency": statistics.mean(e2e),
            "p50_latency": percentile(e2e, 0.50),
            "p90_latency": percentile(e2e, 0.90),
            "p99_latency": percentile(e2e, 0.99),
            "average_ttft": statistics.mean(ttft),
            "p50_ttft": percentile(ttft, 0.50),
            "p99_ttft": percentile(ttft, 0.99),
            "average_tpot": statistics.mean(tpot),
            "average_per_token_latency": statistics.mean(per_token),
            "p99_per_token_latency": percentile(per_token, 0.99),
            "throughput": len(done) / duration_s,
            "output_throughput": sum(r["output_len"] for r in done) / duration_s,
            "input_throughput": sum(r["input_len"] for r in done) / duration_s,
            "duration_s": duration_s,
        },
        "trace": {"path": args.trace, "start_min": args.start_min, "minutes": args.minutes,
                  "slowdown": args.slowdown},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--start-min", type=float, default=0.0, help="window start, trace minutes")
    parser.add_argument("--minutes", type=float, required=True, help="window length, trace minutes")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="stretch inter-arrival times by this factor (lowers the load)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-file", required=True, help="summary, appended as one JSON line")
    parser.add_argument("--requests-file", help="per-request results, one JSON line each")
    args = parser.parse_args()

    results, duration = asyncio.run(replay(args))
    if args.requests_file:
        with open(args.requests_file, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)
    summary = summarize(results, duration, args)
    with open(args.output_file, "a") as f:
        f.write(json.dumps(summary) + "\n")
    s = summary["summary"]
    print(f"{s['completed']}/{s['total_requests']} done in {duration:.0f} s: "
          f"JCT {s['average_latency']:.2f} s (p99 {s['p99_latency']:.2f}), "
          f"TTFT {s['average_ttft']:.2f} s, TPOT {s['average_tpot'] * 1000:.1f} ms")


if __name__ == "__main__":
    main()
