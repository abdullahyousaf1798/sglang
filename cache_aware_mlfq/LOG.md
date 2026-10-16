# Log: cache-aware skip-join MLFQ in SGLang

Raw material for the blog post. One entry per step: what changed and why, how it was tested, and the results.

## Setup used for every result

- **Hardware:** 1x RTX 4090 (24 GB) on RunPod Secure Cloud, EU-RO-1. Host: CUDA 13.0 driver, 46 to 61 GB RAM.
- **Software:** SGLang at commit `bc76a6b` (2026-10-15) plus this branch's changes; PyTorch 2.14.1 with CUDA 13.0; model Qwen3-8B in BF16.
- **Server memory:**
  - The GPU KV pool is fixed at 24,576 tokens (`--max-total-tokens`), so it's the same on any GPU.
  - The CPU-memory KV tier (HiCache, `--enable-hierarchical-cache`) gets 30 GB, which is 203k tokens.
  - Together these make idle conversations spill from GPU to CPU memory.
- **Workload:** SGLang's multi-turn client (`benchmark/hicache/bench_multiturn.py`):
  - 64 conversations of 6 turns each, 384 requests in total.
  - Each turn adds 256 to 1,024 new tokens and gets a 64 to 256 token answer.
  - The next turn carries the full history, so earlier turns become a cached prefix.
  - Requests arrive as a Poisson process at 2 per second, with at most 64 in flight.
  - The average prompt is 2,674 tokens (p90 4,780) and the average answer is 160 tokens.
- **Metrics:** job completion time is each request's end-to-end latency (FastServe's headline metric). Cache sources come from the server's `sglang:prefill_effective_tokens_total` counter.

## Step 1: environment

No SGLang changes.
- The pod runs SGLang from a snapshot of the built environment, restored to local disk on each start in about 15 s.
- An idle watcher stops the pod after 20 minutes without work.
- One finding worth a sentence in the blog: importing SGLang from RunPod's network volume took 319 s, because Python reads tens of thousands of small files, against 12 s from local disk.

## Step 2: benchmark harness and baselines

**What was built** (all in `cache_aware_mlfq/`, no SGLang changes):
- `run_experiment.py` runs one configuration. It starts the server with the shared setup plus that row's scheduling flags, then runs the multi-turn client and saves the client summary and the server's cache counters.
- `make_table.py` builds the comparison table.
- `profile_prefill.py` measures prefill latency over (uncached tokens, cached prefix tokens) for step 3's placement.

**How it was tested:** a pilot fcfs run checked two things.
- **Spilling happens:** 71% of prompt tokens were served from the CPU tier.
- **Queueing dominates latency:** time to first token is 11.2 of the 14.8 s average job completion time, and the waiting queue peaked at 45 requests while about 6 run at a time.

The pilot used exactly the fcfs configuration, so it is the fcfs row.

**Results:** SGLang's existing policies, one run each.

| Run | Job completion, mean (s) | Job completion, p99 (s) | TTFT, mean (s) | TTFT, p99 (s) | TPOT, mean (ms) | Throughput (req/s) | Prompt tokens from GPU cache | Prompt tokens from CPU cache | Preemptions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs | 14.84 | 40.69 | 11.19 | 36.51 | 22.9 | 1.51 | 3% | 71% | 0 |
| lpm | 12.18 | 119.23 | 8.56 | 115.24 | 22.7 | 1.49 | 23% | 52% | 0 |
| shortest-prefill-first | 12.47 | 121.31 | 8.84 | 119.32 | 22.8 | 1.50 | 22% | 53% | 0 |
| hrrn | 14.45 | 50.32 | 10.81 | 45.79 | 22.8 | 1.50 | 3% | 72% | 0 |

**Reading:**
- **Cache-aware ordering helps the mean but hurts the tail.** lpm and shortest-prefill-first cut mean job completion time by about 17% over fcfs, but triple the p99: some requests starve, because these policies never age a request or preempt a running one.
- **They also keep more of the cache on the GPU.** They favor requests whose prefix is still there, so GPU cache hits rise from 3% to about 23%.
- **hrrn adds aging.** That keeps its tail closer to fcfs, but its mean barely improves.
- **The gap step 3 targets:** the mean gain of cache-aware ordering without the tail blow-up. MLFQ's starvation promotion and preemption are designed for exactly that.
- **Caveat:** these are single runs. Step 6 repeats them across several request rates.

**Prefill latency profile** (`prefill_profile_4090.json`, 53 grid points):
- **Method:** each point is the median of 3 timed requests that add fresh tokens on top of a cached random prefix and generate one token. The server confirmed the full prefix was a cache hit every time.
- **Overhead:** request overhead with a 1-token prompt is 19.8 ms. Times below include it.

| Cached prefix (tokens) | 256 new tokens | 1,024 new | 4,096 new | 8,192 new |
| --- | --- | --- | --- | --- |
| 0 | 31 ms | 108 ms | 459 ms | 985 ms |
| 1,024 | 34 ms | 113 ms | 478 ms | 1,022 ms |
| 4,096 | 37 ms | 125 ms | 526 ms | 1,119 ms |
| 8,192 | 41 ms | 140 ms | 590 ms | 1,250 ms |
| 16,384 | 51 ms | 172 ms | 716 ms | n/a (over pool) |

**Reading:**
- **Prefill time grows almost linearly in new tokens,** at about 0.1 ms per token here.
- **A long cached prefix makes the same prefill noticeably slower.** 1,024 new tokens take 108 ms with no cached prefix but 172 ms (+60%) on top of a 16k-token cache hit, because attention still reads the whole prefix.
- **So placement can't use uncached length alone.** It has to use both lengths, which is the cost model step 3 uses.
