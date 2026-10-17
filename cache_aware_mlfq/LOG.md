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
