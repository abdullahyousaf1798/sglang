# Design notes: how the scheduler works, what was dropped and why (2026-10-19)

A plain-language reference for the final design. The results are in `STATUS.md` and `REVIEW.md`; this file explains the mechanism, with one worked example, and records the reasoning behind every cut.

## 1. The two places a request can be

- **Waiting queue:** every request that needs a prefill step next. New arrivals, and retracted requests (their prefill is one token, because their KV is cached).
- **Running batch:** every request that is decoding. They all decode together in one forward pass; nothing can single one out except taking its memory away.
- **One scheduler step** is either one prefill batch or one decode step for the whole running batch. SGLang does not mix the two by default (see section 5 for why, and why we left it that way).

## 2. One step, in order

1. **Sort the waiting queue** (`SchedulePolicy.calc_priority`, then `SkipJoinMlfq.order`). A new request is placed once: its level is the smallest whose quantum (10 ms at level 0, doubling per level) covers the predicted time of its first prefill step. The prediction is from the uncached suffix and the cached prefix (`--sj-mlfq-cost cache-aware`) or from the whole prompt (`prompt`, FastServe's rule). Waiting requests unserved for 3 s are promoted one level. The queue is sorted by (level, time of entry into the level). The level is written to `req.priority = -level`, so SGLang's own priority machinery follows it.
2. **The gate** (`--sj-mlfq-prefill-gate`, `Scheduler._get_new_batch_prefill_raw`): if the head of the queue is less urgent than the most urgent running request, no prefill batch is built this step. A chunked prefill that has already started is always continued (section 6, the known gap).
3. **Admission** (`PrefillAdder`): requests are taken from the head while they fit three limits: the chunk budget (8,192 new tokens per step, 16,384 per batch), the KV budget (free plus evictable cache, minus a reservation for every running request's remaining output), and the running-request cap. A prompt longer than the chunk is split over steps; its finished chunks stay locked in the cache between steps. With the gate, admission also stops at the first request less urgent than the gate level.
4. **Prefill or decode.** If a prefill batch was built it runs alone and every running request waits, KV untouched. Otherwise one decode step runs for all running requests, 30 to 50 ms on this trace.
5. **Charging** (`SkipJoinMlfq.charge`, from `batch_result_processor`): every request in the batch that ran is charged the step's wall time; past its quantum it drops one level. Waiting requests are not charged.
6. **Memory, two separate pools:**
   - **Cache KV** (unlocked; finished turns and waiting requests' prefixes): evicted reactively as space is needed, least recently used first. HiCache has already copied it to CPU (write-through, asynchronous, on a separate stream), so eviction is a drop, not a copy.
   - **Running requests' KV** (locked): taken only by **retraction**, when a decode step cannot allocate its next token and the cache pool is empty. With `--retraction-policy priority` the lowest level goes first. With `--preempt-keep-kv` its KV stays in the cache (CPU-backed) and it re-enters the queue at its level and entry time; it resumes without recompute.
   - **Load-back is reactive:** a returning request whose prefix is on CPU starts the copy at admission and the prefill batch waits for it layer by layer.

## 3. A worked example

Pinned pool of 160k tokens. **A**: returning agent turn, 60k prompt, 55k cached, 30 output tokens. **B**: fresh 50k prompt, 500 output tokens. **C**: fresh 3k prompt, 20 output tokens.

- **A is running, B arrives.** A is level 7 (5k uncached over 55k cached, about 1.2 s predicted). B is level 9 (about 3.8 s). The gate holds B: A keeps decoding at 30 to 50 ms a step and finishes in about 1.5 s. Then B's prefill starts. Without the gate, SGLang starts B at once and A's 30 decode steps each wait behind a 0.7 to 1 s chunk: A takes about 25 s. This is the whole mean-JCT gain; half the requests on the trace are A-like (median output 30 tokens).
- **A and B arrive together.** Nothing is running, the gate is open, A and B's first chunk go into one batch. B's remaining chunks are continued every step and A waits for them. The gate cannot help here; see section 6.
- **C arrives while B is chunking.** C is level 5. With chunk yield (`--sj-mlfq-chunk-yield`), B's chunk is capped so C's prefill shares the batch and C gets its first token now. C's decode still waits for B's remaining chunks. Measured: better TTFT, no mean-JCT change.
- **The pool fills during decode.** Cache KV is dropped first, LRU. When nothing is left, the lowest-level running request is retracted with its KV kept; it is first in line at its level when memory frees. An urgent waiting request never evicts a runner (preemption is off); it waits, and the gate keeps less-urgent prefills from jumping ahead of it.

## 4. FastServe versus this port

| | FastServe | This port |
| --- | --- | --- |
| Scheduling unit | per iteration, per job | per step, whole batch (prefill or decode) |
| Pausing a job | free: not picked next iteration | only by releasing its memory (retraction) |
| Prefix cache | none; finished means freed | radix cache; finished KV stays until evicted |
| Placement cost | full prompt | uncached suffix over cached prefix |
| Swapping | proactive, by estimated next run time, paused jobs only | reactive eviction of cache KV; retraction of runners as last resort |
| Preemption for an urgent arrival | yes, cheap | off: it churned long requests (p99 up to 318 s) |

## 5. What was tried and dropped, with the evidence

- **Mixed prefill and decode steps** (`--enable-mixed-chunk`): decodes ride inside each prefill chunk, so TPOT falls 5x, but a step still lasts a chunk, so a 30-token answer still takes 30 long steps: mean JCT unchanged at 8k chunks. At 2k chunks the step count quadruples and each step re-pays attention over long contexts: throughput -12%, JCT +68%. At saturation even 8k mixed steps slowed prefill enough to cost 8 to 38%. Decision: the final design does not depend on mixing; the gate removes the cause instead of smoothing the symptom.
- **Priority preemption** (threshold 0 or 3): every setting churned, 38 to 111 preemptions per run, long requests' p99 doubled. Off.
- **Two-tier eviction** (`--radix-eviction-policy mlfq`): orders only the cache pool, finished turns before waiting requests' prefixes, then by level. LRU already does the first part because the prefix match touches every waiting request's prefix each step; the second part is reached only when the cache holds nothing but waiting prefixes, worth about one second of reloads per five-minute run here. No measurable effect. Implemented, not in the PR.
- **Proactive load-ahead** (`--sj-mlfq-load-ahead`): without preemption, reload volume is about a second per run; nothing to hide. Implemented, not in the PR.
- **Chunk yield** (`--sj-mlfq-chunk-yield`): TTFT -10%, p50 -5%, mean 0%, long p99 +4%, TPOT worse. Not in the PR; documented follow-up.
- **Profile file** (`--sj-mlfq-profile`): replaced by `OnlinePrefillCostModel`, which fits a compute scale and an attention scale to the prefill batches the scheduler runs. Still accepted for experiments.
- **Full-prompt placement** (`--sj-mlfq-cost prompt`): kept only as the ablation; cache-aware is 9% better on the mean and 25% better for returning turns.

## 6. Known gaps and open questions

1. **Chunk pause.** The gate decides whether to start a prefill, not whether to continue a started chunk. A chunked prefill in progress starves more urgent decodes until it finishes. The chunk's KV stays locked between steps, so pausing it is cheap: apply the gate to the continuation too. This is the first follow-up to measure.
2. **Load time is not in the placement cost.** A 55k-token prefix is about 8 GB, about 0.3 s of copy, against 1.2 s of compute; levels double, so it rarely changes the level. Cheap to add as a per-host-token term.
3. **Saturation, window 25:** shortest-prefill-first beats ours by 6% at slowdown 11. Not understood; one run.
4. **Yield's TPOT regression** (69 to 189 ms) is not explained. Probably the capped chunk admits more decoders per step.
5. **Promotion is checked every scheduling step** in the final code, not every 1,000 steps as in FastServe; the 3 s threshold is FastServe's.
6. **PD disaggregation:** the decode GPU never prefills, so the stall this fixes does not exist there; placement and ordering still apply on the prefill node. The claim is for single-engine serving.
7. **Single model and GPU type** (Qwen3-8B with YaRN on A100 80 GB). A second model size is the next evidence to buy.

## 7. The upstream bug that crashed rows

`PrefillBudget.available_chunk_tokens` gave a continuing chunk its full size "to make progress" even when the pool physically held nothing free and nothing evictable; `alloc_token_slots` then failed and the server died ("Out of memory. Try to lower your batch size."). It bites any policy that packs requests beside a chunk until the pool is 100% locked, shortest-prefill-first included. Fix (`prefill_budget.py`, `scheduler.py`): bound the chunk by physically available tokens, return None when starved, and skip the prefill step so a decode step retracts. Regression check in `test_skip_join_mlfq.py`. Worth its own upstream PR.
