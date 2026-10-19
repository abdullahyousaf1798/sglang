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

## Step 3: skip-join MLFQ, plain and cache-aware

**What changed in SGLang** (8 files, about 110 lines plus one new module):

| File | Change |
| --- | --- |
| `managers/skip_join_mlfq.py` (new) | Cost model built from the profile, plus placement, ordering, demotion and starvation promotion. |
| `managers/schedule_batch.py` (`Req`) | Four per-request fields: level, service time, level entry time, last service time. |
| `managers/schedule_policy.py` (`SchedulePolicy`) | New `sj-mlfq` policy; `calc_priority` calls `mlfq.order(waiting_queue)`. |
| `managers/scheduler_components/batch_result_processor.py` | After each batch, charges every request in it `now - batch.launch_ts`. |
| `managers/scheduler.py` | Memory-full preemption in the admission loop. Preemptions are counted on the retraction metrics. |
| `arg_groups/fields/schedule.py`, `validation_hook.py` | New flags `--sj-mlfq-cost {prompt,cache-aware}` and `--sj-mlfq-profile`. Priority scheduling is now allowed with `sj-mlfq`. |

The level is stored as `req.priority = -level`, so SGLang's existing priority sorting, preemption and retraction follow MLFQ order. Quanta are FastServe's: 10 ms doubling per level, and promotion after 3 s without service, checked every 1,000 steps.

**Placement**, the core of the idea:

```python
prompt_len = len(req.origin_input_ids)
if self._cache_aware:
    cached = req.num_matched_prefix_tokens  # GPU-resident plus CPU-resident (HiCache) prefix hits
    cost_s = self._cost_model.predict_s(uncached=prompt_len - cached, cached=cached)
else:  # FastServe: a fresh prefill of the whole prompt
    cost_s = self._cost_model.predict_s(uncached=prompt_len, cached=0)
level = 0 if cost_s <= BASE_QUANTUM_S else math.ceil(math.log(cost_s / BASE_QUANTUM_S, QUANTUM_MULTIPLIER))
```

Example: a returning turn with a 5,000-token prompt, of which 4,500 tokens are cached, goes to level 6 (640 ms quantum) under FastServe's rule and level 3 (80 ms) under cache-aware placement.

**What porting FastServe to SGLang required:**
1. **Preemption on full memory.** SGLang's priority preemption only fired when the batch hit its request-count limit (2,048 here, never reached). A memory-bound server therefore never preempted, so MLFQ demotion had no effect. The admission loop now lets a request that doesn't fit preempt lower-priority running requests:

    ```python
    if (res == AddReqResult.NO_TOKEN
            and self.enable_priority_preemption
            and not (adder.can_run_list and req is adder.can_run_list[-1])
            and adder.preempt_to_schedule(req)):
        req.init_next_round_input(self.tree_cache)  # preemption can evict tree nodes
        res = adder.add_one_req(req, ...)
    ```

2. **A bug from misreading `NO_TOKEN`.** `NO_TOKEN` also follows a successful admission that used up the budget. The first version treated it as "didn't fit" and admitted such requests a second time. That caused two crashes: a preemption hit a request whose KV was already freed, and SGLang's pool audit found a 27-token leak. The `can_run_list` check above fixes it.
3. **Preemption discards work in SGLang.** Its preemption frees the preempted request's decoded tokens, which must be recomputed. FastServe instead keeps them by swapping to CPU memory. Step 4 closes this gap.
4. **Placement only matters when memory is short.** SGLang batches are limited by memory, not by a batch-size cap. So MLFQ decides who runs only when memory is short; GPU KV usage stayed around 88-96% in these runs.

**How it was tested:**
- Five unit checks on the Mac (`test_skip_join_mlfq.py`): the cost model against the profile, placement levels, demotion, ordering, and starvation promotion.
- Both rows ran to completion on the GPU with no errors.

**Results** (one run each):

| Run | Job completion, mean (s) | Job completion, p99 (s) | TTFT, mean (s) | TTFT, p99 (s) | TPOT, mean (ms) | Throughput (req/s) | Prompt tokens from GPU cache | Prompt tokens from CPU cache | Preempted or retracted | Decoded tokens recomputed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs | 14.84 | 40.69 | 11.19 | 36.51 | 22.9 | 1.51 | 3% | 71% | 0 | 0 |
| lpm | 12.18 | 119.23 | 8.56 | 115.24 | 22.7 | 1.49 | 23% | 52% | 0 | 0 |
| shortest-prefill-first | 12.47 | 121.31 | 8.84 | 119.32 | 22.8 | 1.50 | 22% | 53% | 0 | 0 |
| hrrn | 14.45 | 50.32 | 10.81 | 45.79 | 22.8 | 1.50 | 3% | 72% | 0 | 0 |
| sj-mlfq-prompt | 16.35 | 65.04 | 4.74 | 21.80 | 68.5 | 1.62 | 3% | 72% | 310 | 28,081 |
| sj-mlfq-cache-aware | 12.96 | 64.95 | 2.49 | 25.32 | 65.1 | 1.60 | 6% | 68% | 233 | 20,435 |

Levels assigned at placement:
- **sj-mlfq-prompt:** {0: 2, 1: 10, 2: 14, 3: 40, 4: 54, 5: 107, 6: 147, 7: 1}, with 2,353 demotions and 158 starvation promotions.
- **sj-mlfq-cache-aware:** {0: 2, 1: 22, 2: 85, 3: 172, 4: 84, 5: 2, 6: 3}, with 2,715 demotions and 111 promotions.

**Reading:**
- **Cache-aware placement beats FastServe's placement on every latency mean.**
  - Job completion time is 21% lower (16.35 to 12.96 s), and time to first token is 47% lower (4.74 to 2.49 s).
  - It causes 25% fewer preemptions and 27% less recomputation.
  - Why: FastServe's rule charges a returning turn for its whole history, so most requests land at levels 5-6. Cache-aware placement puts them at levels 2-4, which matches their real prefill cost.
- **Against fcfs, cache-aware SJ-MLFQ's main win is first-token latency.** Time to first token drops 78% (11.19 to 2.49 s) and mean job completion time drops 13%. Against lpm, the best baseline mean, it roughly halves the p99 (65 vs 119 s).
- **The cost is decode speed.** Per-token decode time is about 3x fcfs, and the p99 is above fcfs. Both come from preemption throwing away decoded KV: about 20k of roughly 61k generated tokens were recomputed. That is the cost FastServe avoids by swapping instead of discarding, and the cost Step 4 removes.
- **Caveat:** single runs. Step 6 repeats them.

## Step 4a: keep preempted KV

**What changed** (3 SGLang lines plus a flag):
- **`managers/schedule_batch.py`, `release_req`.** This is the only place a preempted or retracted request gives up its KV. With `--preempt-keep-kv`, it now checkpoints that KV into the radix tree instead of freeing it. This resolves a TODO that was already in the code ("for preempted requests, we may want to insert into the tree").

    ```python
    # before
    release_kv_cache(req, tree_cache, checkpoint=False)
    # after
    release_kv_cache(req, tree_cache, checkpoint=get_schedule().preempt_keep_kv)
    ```

    The preempted request's prompt and decoded tokens become an unlocked cached prefix. Under pressure, HiCache moves it to CPU memory rather than deleting it, which is FastServe's swap-out. When the request returns, its prefix match finds all of it.
- **Harness.** It now records all prefill compute (`sglang:realtime_tokens_total{mode="prefill_compute"}`). Subtracting first-attempt compute gives the tokens re-prefilled after preemption.

**Results** (one run):

| Run | Job completion, mean (s) | Job completion, p99 (s) | TTFT, mean (s) | TTFT, p99 (s) | TPOT, mean (ms) | Throughput (req/s) | Preempted or retracted | Decoded tokens at preemption | Prefill tokens recomputed after preemption |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sj-mlfq-cache-aware | 12.96 | 64.95 | 2.49 | 25.32 | 65.1 | 1.60 | 233 | 20,435 | not recorded (about 20k) |
| sj-mlfq-cache-aware-keepkv | 14.03 | 60.29 | 3.88 | 27.60 | 61.3 | 1.63 | 218 | 19,576 | 218 |

**Reading:**
- **The mechanism works.** 218 preemptions cost 218 recomputed tokens, one per resume (the newest token). Everything else came back from the GPU or CPU cache.
- **Latency did not improve.** Mean job completion time and TTFT are slightly worse, while p99 and TPOT are slightly better. Single runs, so differences of about 1 s may be noise.
- **The cause is that recompute was never the bottleneck.** About 20k recomputed tokens cost about 2 s of GPU time over a 5-minute run. The dominant cost is churn: about 3,000 demotions and more than 200 preemptions. With FastServe's 10 ms base quantum and roughly 25 ms decode steps here, a request is demoted every few steps. Each new arrival then outranks, and preempts, a request partway through decoding.

## Step 4 sweep: preemption threshold, base quantum, and run-to-run noise

**What changed:**
- **`skip_join_mlfq.py`:** the base quantum is now a constructor argument (default: FastServe's 10 ms).
- **New flag:** `--sj-mlfq-base-quantum-ms`.
- **New script:** `run_step4_sweep.sh`, which runs the five rows below.

Every MLFQ row uses cache-aware placement with preempted KV kept.

**Why these knobs:**
- **The preemption threshold** sets how many levels better a waiting request must be before it can push out a running one. With threshold 0, any newcomer evicts a request that has used up even its first quantum.
- **The base quantum matters less than it first appears.** Skip-join gives a request the smallest quantum that covers its prefill, whatever the base. A typical 90 ms prefill starts with a 160 ms quantum at a base of 10, 40 or 160 ms, then doubles on the same schedule. The base mostly renumbers the levels.

**Results** (one run each):

| Run | Job completion, mean (s) | Job completion, p99 (s) | TTFT, mean (s) | TTFT, p99 (s) | TPOT, mean (ms) | Throughput (req/s) | Preempted or retracted | Demotions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs | 14.84 | 40.69 | 11.19 | 36.51 | 22.9 | 1.51 | 0 | n/a |
| fcfs-run2 | 14.42 | 38.00 | 10.79 | 33.89 | 22.8 | 1.50 | 0 | n/a |
| lpm | 12.18 | 119.23 | 8.56 | 115.24 | 22.7 | 1.49 | 0 | n/a |
| keepkv (threshold 0) | 14.03 | 60.29 | 3.88 | 27.60 | 61.3 | 1.63 | 218 | about 3,000 |
| keepkv-run2 (threshold 0) | 16.29 | 70.42 | 3.79 | 24.19 | 74.7 | 1.62 | 292 | 3,277 |
| keepkv-q40 (threshold 0, 40 ms base) | 15.34 | 61.22 | 4.54 | 26.68 | 62.4 | 1.62 | 252 | 2,797 |
| keepkv-gap1 (threshold 1) | 13.12 | 55.45 | 2.52 | 22.34 | 65.8 | 1.58 | 206 | 2,772 |
| keepkv-gap2 (threshold 2) | 13.78 | 59.55 | 4.95 | 29.04 | 53.2 | 1.64 | 198 | 2,707 |

**Reading:**
- **MLFQ is much noisier than fcfs.** Repeats of fcfs differ by 3% in mean job completion time, while repeats of the same MLFQ configuration differ by 15% (14.0 vs 16.3 s). Preemption dynamics amplify small timing differences. Gaps of about 2 s between MLFQ variants are within noise, so the blog needs at least 3 repeats per configuration.
- **The base quantum barely matters, as predicted.** At 40 ms, placement shifts to levels 0-2, but demotions and latency stay within noise.
- **The threshold trades as expected, with small effects.** Threshold 2 trades first-token latency (4.95 s) for decode speed (53 ms per token). Threshold 1 gives the best MLFQ numbers in this sweep, but within noise.
- **On this synthetic workload, no MLFQ setting reliably beats lpm on mean job completion time.** Every setting does cut mean TTFT by 55-77% against fcfs and lpm, and roughly halves lpm's p99. Each also decodes 2-3x slower per token than fcfs.
- **Next:** this workload's output lengths are uniform (64-256 tokens), which is where MLFQ's least-attained-service ordering helps least. The next step moves the main workload to real ShareGPT conversations, with heavy-tailed answer lengths, as in FastServe's own evaluation.

## Main table: real ShareGPT conversations, 3 seeds

**Setup:**
- 200 real multi-turn ShareGPT conversations (`prepare_sharegpt.py`), each reply as long as the real one, released at 3.0 requests per second.
- A pilot put capacity at about 2.8 requests per second, so this is at saturation.
- 3 client seeds per configuration (`run_sharegpt_main.sh 3.0 200`). Cells are mean ± standard deviation over seeds.

| Config | Job completion, mean (s) | Job completion, p99 (s) | TTFT, mean (s) | TTFT, p99 (s) | TPOT, mean (ms) | Throughput (req/s) | Preemptions per run | Load-back copy time per run (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs | 11.58 ± 2.88 | 32.72 ± 7.48 | 5.21 ± 2.55 | 23.33 ± 7.64 | 23.1 ± 0.2 | 2.49 ± 0.10 | 0 | 3.9 ± 0.4 |
| lpm | 11.36 ± 2.65 | 92.34 ± 34.34 | 5.00 ± 2.31 | 87.19 ± 33.88 | 23.1 ± 0.3 | 2.47 ± 0.11 | 0 | 2.3 ± 0.1 |
| shortest-prefill-first | **10.03** ± 1.92 | 93.26 ± 33.02 | 3.62 ± 1.58 | 81.88 ± 31.89 | 23.3 ± 0.3 | 2.52 ± 0.13 | 0 | 3.3 ± 0.4 |
| sj-mlfq-prompt (FastServe) | 16.72 ± 4.74 | 95.59 ± 15.74 | 0.12 ± 0.02 | 0.87 ± 0.41 | 52.5 ± 12.2 | 2.33 ± 0.17 | 1,920 ± 608 | 27.6 ± 8.2 |
| sj-mlfq-cache-aware | 17.16 ± 5.20 | 100.47 ± 20.65 | 0.11 ± 0.01 | 0.63 ± 0.30 | 53.7 ± 13.1 | 2.31 ± 0.15 | 1,971 ± 640 | 28.4 ± 9.4 |
| sj-mlfq-cache-aware-keepkv | 16.49 ± 4.99 | 94.13 ± 25.49 | **0.10** ± 0.02 | 0.79 ± 0.47 | 52.2 ± 12.4 | 2.32 ± 0.17 | 1,882 ± 643 | 25.0 ± 9.5 |

**Reading:**
- **First tokens are dramatically faster.** Every MLFQ variant cuts mean time to first token from about 5 s to about 0.1 s, and p99 from 23-87 s to under 1 s.
- **Everything else gets worse.**
  - Mean job completion time is 42-48% worse than fcfs, and 64-71% worse than shortest-prefill-first.
  - Decoding is 2.3x slower per token.
  - Throughput is 7% lower.
- **Job completion time is the wait plus the answer, and MLFQ moves the wait into the answer.**
  - fcfs averages 5.2 s of waiting before the first token and 6.4 s generating the answer.
  - MLFQ waits 0.1 s but spends 16.4 s on the answer. With memory full, each new request starts in the top queue and pauses a running request that has already been demoted. That is about 2.7 pauses per request (about 700 requests per run).
- **The pauses are not free.**
  - Each paused request later reloads its KV from CPU memory: 25-28 s of copy time per run against 2-4 s for the baselines.
  - Each new or resumed request's prefill stalls the whole decode batch.
  - The lost capacity, at a load already at saturation, turns into waiting for everyone.
- **Reordering by size does help when it is free.** Shortest-prefill-first never preempts, and it beats fcfs on mean job completion time by 13%.
- **Cache-aware placement and keep-KV are within noise of FastServe's placement here.** Histories average about 1.2k tokens and every prefill is cheap, so the placement rule has little to distinguish. Recompute was never the dominant cost.

**Why this workload is close to the worst case for both ideas, and what FastServe actually measured:**
- **FastServe's evaluation** used single-turn ShareGPT and Alpaca requests with Poisson arrivals: no conversations and no prefix cache. It ran OPT models on 40 GB A100s against 2023 vLLM (v0.1.7: first-come-first-served, run to completion, no chunked prefill or prefix caching).
- **Its main metric** was average latency per output token (end-to-end time divided by output length) and throughput under a 0.3 s-per-token target. That metric weighs short requests that waited heavily, which is exactly what MLFQ fixes.
- **Its gains** were largest with long, skewed jobs at high load. On Alpaca's short jobs every system stayed fast. Swapping cost under 5% of latency because it was proactive.
- **What our ideas need:**
  1. Long, mostly cached histories, so the full prompt and the uncached suffix put a request in very different queues.
  2. Very uneven answer lengths, so letting short requests pass long ones pays off.
  3. Cheap preemption: keep-KV, two-tier eviction (step 4b) and proactive load-back (step 5).
- **This workload has none of the three.** Short histories, moderate answers, and preemption that costs 7% of throughput.

**Next workload candidate: Mooncake production traces** (Kimi, FAST'25). These are one-hour traces with real arrival times, lengths and 512-token prefix-block sharing:

| Trace | Requests in 1 h | Input tokens: mean / median / p99 | Output tokens: mean / median / p90 | Prompt shared with an earlier request |
| --- | --- | --- | --- | --- |
| conversation | 12,031 | 12.0k / 6.9k / 85k | 343 / 350 / 597 | 37% |
| tool & agent | 23,608 | 8.6k / 6.3k / 62k | 182 / 30 / 507 | 57% |

The tool-and-agent trace meets all three conditions. It needs an 80 GB GPU: the 4090's KV pool holds less than one p99 prompt. It also needs a replayer that preserves the block sharing, because SGLang's built-in Mooncake replay shrinks each block to about 130 filler tokens.

## Steps 4b, 5 and 6 on Mooncake production traces (A100 80 GB)

**Setup:**
- **Hardware:** two A100 SXM 80GB pods, each running every row of its trace windows, so comparisons within a window are on one machine.
- **Model:** Qwen3-8B with Qwen's documented YaRN extension to 131k. Trace prompts reach 126k tokens; SGLang also needed `max_position_embeddings` raised to 131072.
- **Workload:** the Mooncake tool-and-agent trace. Four 0.4-minute windows at trace minutes 10, 25, 40 and 55, chosen before any results; about 180 requests each.
  - The replay keeps the trace's prefix sharing: each 512-token block id becomes a fixed block of token ids (`mooncake_replay.py`).
  - Arrival times are stretched 11x (slowdown 11, SGLang's own replay method). At slowdown 8, fcfs served only 0.84 of the 1.03 requests per second that arrived; 11 puts the load near 85% of capacity.
- **Profile:** a new A100 prefill profile, 62 cells up to 32k new tokens on 96k cached.
- **Two memory settings:**
  - **Natural pool:** whatever fits, 393k tokens. Windows 10 and 25 only.
  - **Pool pinned at 160k tokens:** all 4 windows. This is the tightest round size that still serves the largest request. SGLang's own HiCache benchmark constrains memory the same way (`bench_mix.sh`, `--mem-frac 0.25`).
- **New mechanisms, all opt-in:**
  - **Step 4b, two-tier eviction** (`--radix-eviction-policy mlfq`): unclaimed KV, such as finished conversations, is evicted first by LRU; then waiting and preempted requests' KV, lowest level first.
  - **Step 5, load-ahead** (`--sj-mlfq-load-ahead`): the CPU-cached prefixes of the next 8 waiting requests, in MLFQ order, are copied back while other batches run. It extends FastServe's proactive swap-in to returning turns' cached history.
  - **Step 6, the prefill gate** (`--sj-mlfq-prefill-gate`): SGLang runs a decode step instead of a new prefill while every running request is in a higher queue than the next waiting one. This is FastServe's iteration-level priority fitted to SGLang's separate prefill and decode steps.
- **Rows:** a 2x2 of placement (FastServe's full-prompt cost, cache-aware) by eviction (SGLang's LRU, two-tier). Every cell has FastServe's own mechanisms ported (kept KV, load-ahead), and every cell runs with and without the gate:

  | Row | Placement | Eviction |
  | --- | --- | --- |
  | `fastserve-lru` | FastServe | LRU (this is FastServe ported) |
  | `ours-lru-eviction` | cache-aware | LRU |
  | `fastserve` | FastServe | two-tier |
  | `ours` | cache-aware | two-tier |

**Bugs found along the way:**
- **The first gate checked only the head of the waiting queue.** A short, urgent head let long low-priority prompts into the same prefill batch, so the gate never held anything back and matched `fastserve` to 0.01 s. It now stops admission at the first request less urgent than the most urgent running one.
- **The first `fastserve` rows had two-tier eviction on.** That is one of the two changes being measured, so they were not a FastServe baseline; `fastserve-lru` was added as the faithful port.
- **SGLang's experimental `shortest-prefill-first` crashed** with "Out of memory. Try to lower your batch size" on windows 10 and 55 of the pinned pool. That is an upstream bug; it is reported over the 2 windows it finished.

### Natural pool (393k tokens), windows 10 and 25

Change against fcfs on the same windows (geometric mean of per-window ratios; negative is better):

| Config | JCT mean | JCT p50 | JCT p99 | TTFT | TPOT | Per-token latency | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs (absolute) | 20.68 s | 13.57 s | 76.5 s | 6.90 s | 1076 ms | 1444 ms | 0.69 req/s |
| lpm | +4% | +2% | +2% | -26% | +50% | +4% | 0% |
| shortest-prefill-first | +1% | 0% | 0% | -57% | +64% | -1% | 0% |
| sj-mlfq-naive | -1% | 0% | 0% | -43% | +42% | -1% | 0% |
| fastserve | -1% | 0% | -1% | -48% | +50% | -1% | 0% |
| ours | -1% | 0% | -1% | -48% | +54% | -1% | 0% |
| **fastserve-gate** | **-35%** | **-52%** | -12% | -16% | **-88%** | **-57%** | 0% |
| **ours-gate** | **-34%** | **-52%** | -11% | -19% | **-88%** | **-61%** | -1% |

### Pool pinned at 160k tokens, 4 windows

| Config | JCT mean | JCT p50 | JCT p99 | TTFT | TPOT | Per-token latency | Throughput | Preempted per run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs (absolute) | 23.37 s | 18.08 s | 78.8 s | 14.10 s | 453 ms | 1562 ms | 0.66 req/s | 0 |
| lpm | -13% | -18% | 0% | -28% | +18% | -43% | +1% | 0 |
| shortest-prefill-first (2 windows) | -16% | -13% | -5% | -57% | +106% | -34% | +1% | 0 |
| fastserve-lru (FastServe ported) | -11% | -15% | +15% | -48% | +81% | -29% | 0% | 20 |
| ours-lru-eviction | -15% | -14% | +5% | -49% | +98% | -31% | 0% | 12 |
| fastserve | -8% | -14% | +13% | -47% | +64% | -30% | 0% | 25 |
| ours | -18% | -20% | +9% | -51% | +62% | -39% | +1% | 14 |
| fastserve-lru-gate | -21% | -32% | +30% | -40% | -68% | -53% | -2% | 19 |
| ours-lru-gate | -21% | -38% | +11% | -42% | -76% | -59% | -2% | 34 |
| fastserve-gate | -20% | -43% | +29% | -46% | -72% | -58% | -1% | 27 |
| ours-gate | -20% | -38% | +14% | -46% | -71% | -62% | -2% | 44 |

Mean JCT per window shows where the gate stops paying off:

| Config | Window 10 | Window 25 | Window 40 | Window 55 |
| --- | --- | --- | --- | --- |
| fcfs | 26.40 | 13.33 | 23.55 | 30.18 |
| lpm | 21.59 | 12.91 | 22.23 | 23.35 |
| fastserve-lru | 20.47 | 12.77 | 20.93 | 28.67 |
| ours | 18.05 | 12.65 | 20.65 | 23.60 |
| fastserve-lru-gate | 17.35 | 10.62 | 19.48 | 26.59 |
| ours-gate | 17.31 | 9.65 | 19.49 | 31.34 (111 preemptions) |

### The two changes on top of FastServe (pinned pool, 4 windows)

Change from the row without the change to the row with it:

| Change | Gate | JCT mean | JCT p50 | JCT p99 | TTFT | Per-token latency |
| --- | --- | --- | --- | --- | --- | --- |
| Cache-aware placement (LRU eviction) | off | -5% | +2% | -9% | -3% | -2% |
| Cache-aware placement (two-tier eviction) | off | -11% | -7% | -4% | -8% | -12% |
| Two-tier eviction (FastServe placement) | off | +3% | +2% | -2% | +2% | -2% |
| Two-tier eviction (cache-aware placement) | off | -4% | -7% | +3% | -3% | -12% |
| Both, against FastServe ported | off | -8% | -6% | -5% | -6% | -14% |
| Cache-aware placement (LRU eviction) | on | +1% | -9% | -15% | -3% | -14% |
| Cache-aware placement (two-tier eviction) | on | 0% | +9% | -11% | 0% | -10% |
| Two-tier eviction (FastServe placement) | on | +2% | -16% | -1% | -9% | -12% |
| Two-tier eviction (cache-aware placement) | on | +1% | 0% | +3% | -7% | -8% |
| Both, against FastServe ported + gate | on | +2% | -9% | -12% | -9% | -21% |

Returning turns (at least half the prompt is a prefix an earlier request sent; 47-63% of requests) against fresh prompts, mean over windows:

| Config | Returning: JCT (s) | Returning: TTFT (s) | Fresh: JCT (s) | Fresh: TTFT (s) |
| --- | --- | --- | --- | --- |
| Natural pool, fcfs | 16.57 | 6.88 | 27.58 | 6.93 |
| Natural pool, fastserve-gate | 7.65 | 4.79 | 22.46 | 7.85 |
| Natural pool, ours-gate | 7.81 | 4.27 | 22.76 | 8.14 |
| Pinned pool, fcfs | 18.71 | 14.02 | 28.69 | 14.21 |
| Pinned pool, fastserve-lru-gate | 9.69 | 6.27 | 28.68 | 10.64 |
| Pinned pool, ours-gate | 9.34 | 5.07 | 30.90 | 10.61 |

**Reading:**
- **With plenty of memory, admission order cannot change JCT; only the gate can.**
  - SGLang runs any admissible prefill before the next decode step, and on this trace each prefill chunk takes about 1 s. Every running request freezes for every newcomer's prefill.
  - Reordering the queue (lpm, shortest-prefill-first, MLFQ without the gate) only moves waiting between TTFT and TPOT: mean JCT stays within 4% of fcfs.
  - The gate lets urgent requests finish first: -35% mean and -52% median JCT, -88% TPOT, same throughput. This holds on both windows (26.3 to 16.1 s, 15.0 to 10.25 s).
  - Nothing was preempted or reloaded from CPU here, so 4b and 5 had nothing to act on.
- **Under memory pressure, every MLFQ variant beats fcfs** (-8% to -21% mean), and the gated ones cut the median by 32-43%.
  - **But the heaviest window (55) shows the gate's limit.** Urgent requests the gate admits preempt running ones when memory is short. `ours-gate` preempted 111 requests and ended 4% slower than fcfs; the same churn hurt MLFQ on ShareGPT.
  - Limiting that churn is the open problem: a preemption threshold, or not preempting for gate admissions.
- **Cache-aware placement helps, mostly where FastServe's own metric looks.**
  - Without the gate it cuts mean JCT 5-11% against FastServe's placement.
  - With the gate the mean is unchanged, but per-token latency (end-to-end time over output length, FastServe's metric) drops 10-21%, and p99 drops 11-15% (noisy at about 180 requests per window).
  - Returning turns get their first token 11% sooner with the natural pool and 2-4% sooner with the pinned pool; fresh prompts give up 4-8%.
- **Two-tier eviction shows no consistent gain** (-4% to +3% on the mean). Two likely reasons:
  - LRU already evicts old conversations first.
  - In agent traffic, a turn that just finished is what the same session's next turn reuses seconds later, so evicting finished KV first is not reliably right.
  - Memory pressure was also moderate: 12-44 preemptions per run.
- **Noise:** each window ran once. Differences under about 10% on the mean are within noise.

## Final round: independent review, bug fixes and the final table (2026-10-18 to 19)

**What changed and why.**
- **An independent design review** (`REVIEW.md`) re-derived the design from the evidence. It kept skip-join MLFQ, cache-aware placement, the gate and kept KV; it demanded the missing SGLang-native baselines (mixed chunks, `--prefill-decode-interval`) and proposed an online cost model, chunk yield and churn control.
- **Online cost model** (`OnlinePrefillCostModel`, `skip_join_mlfq.py`): without `--sj-mlfq-profile`, two scale factors (compute per uncached token, attention of new tokens over the cached prefix) are fitted by ridge least squares with forgetting from the prefill batches the scheduler runs. A side-script JSON was a merge blocker.
  > **Before:** `--sj-mlfq-profile` required.
  > **After:** `PrefillCostModel.from_profile(path) if path else OnlinePrefillCostModel()`, charged from `batch_result_processor` after every prefill batch.
- **Chunk yield** (`--sj-mlfq-chunk-yield`, `mlfq_chunk_limit` in `skip_join_mlfq.py`, hooked where shortest-prefill-first's chunk limit is): waiting requests at a better level than the chunked request share its prefill batch.
- **Preemption off** in the final design: thresholds 0 and 3 both churned (38 to 111 preemptions per run, long requests' p99 doubled).
- **Upstream bug fixed** (`mem_cache/prefill_budget.py`, `scheduler.py`): a continuing chunk was handed its full size with the pool 100% locked, and allocation crashed the server. Now a starved chunk skips the step so decode retracts.
  > **Before:** `if rem_tokens <= 0: return self.chunk_size  # make progress`
  > **After:** bound by physically available tokens; `None` when nothing fits, and `_get_new_batch_prefill_raw` returns no batch that step.
- **Harness:** `run_experiment.py` waits for the previous server's processes and 95 GB of free host memory (six pods shared one host). Results moved to a shared network volume after a stopped pod's local disk became unreachable when its host GPU was taken.

**How it was tested.** 12 unit checks on the Mac; two rounds on A100s. Round 1 (0.4-minute windows, two pods): the baselines the review asked for, 2k versus 8k mixed chunks, three preemption settings. Round 2 (the final table): six pods, 1-minute windows 10, 25, 40, 55, slowdown 15 and 11, pinned and natural pools, eight rows, repeats of fcfs and ours, the conversation trace; 75 rows, 4 crashed before the fix and were rerun.

**Results** (pinned 160k pool, slowdown 15, four 1-minute windows; change against fcfs, equal throughput):

| Row | JCT mean | JCT p50 | JCT p99 | TTFT | TPOT |
| --- | --- | --- | --- | --- | --- |
| fcfs | 17.5 s | 14.0 s | 74.2 s | 8.3 s | 646 ms |
| shortest-prefill-first (best native) | -12% | -6% | -17% | -54% | +62% |
| lpm + mixed 8k | -12% | -15% | -1% | -24% | -54% |
| FastServe placement | -24% | -43% | 0% | -28% | -81% |
| **ours** | **-30%** | **-51%** | -1% | -40% | -89% |

Full tables with the saturation, natural-pool and conversation rows: `STATUS.md` section 4 and `results/FINAL_TABLES.md`.

**Reading.** The gate turns SGLang's prefill-first stall into a priority decision, and that is worth 30% of mean JCT on agent traffic with no loss for long requests; cache-aware placement adds a consistent 9% over FastServe's placement, all of it on returning turns. Mixed chunks, smaller chunks, preemption, two-tier eviction, load-ahead and chunk yield each either did nothing or hurt, and the table says so.

**Lessons for the write-up.** Compare against SGLang's own best configuration, not fcfs; report long requests' tail; calibrate the slowdown on 1-minute windows (11 is saturation there); keep results on a network volume.
