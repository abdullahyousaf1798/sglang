# Cache-aware skip-join MLFQ for SGLang

Research branch behind the post [Rethinking LLM Scheduling in SGLang: Cache-Aware MLFQ and Prefill Gating](https://abdullahyousaf.com/rethinking-llm-scheduling-in-sglang/) and the pull requests listed at the end.

The policy itself is engine code: `python/sglang/srt/managers/skip_join_mlfq.py` plus small hooks in the scheduler, the schedule policy, the batch result processor and the prefill budget. This folder holds everything around it: the trace replayer, the benchmark harness, the pod scripts, every result row, and the notes.

## What the policy does

`--schedule-policy sj-mlfq` is FastServe's skip-join multi-level feedback queue as an SGLang scheduling policy, with two additions:

- **Cache-aware placement.** A request's starting queue level comes from the predicted cost of prefilling its uncached tokens on top of its cached prefix, not from its full prompt length. The prediction is an online cost model fitted from the prefill batches the scheduler runs; no profile file is needed.
- **Prefill gate.** No new prefill batch is started while a request that is already generating is more urgent than the best waiting request; a decode step runs instead. A chunked prefill that has already started continues.

Everything is opt-in. The configuration measured below is:

```
--schedule-policy sj-mlfq --enable-priority-scheduling --retraction-policy priority \
--preempt-keep-kv --sj-mlfq-prefill-gate --sj-mlfq-cost cache-aware --disable-priority-preemption
```

Two things tried on this branch and left out of the pull requests, because they did not help on this workload: proactive KV load-back (`--sj-mlfq-load-ahead`), two-tier eviction (`--radix-eviction-policy mlfq`), chunk yield (`--sj-mlfq-chunk-yield`) and chunk pause (`--sj-mlfq-chunk-pause`). `DESIGN.md` has the reasons and numbers.

## Results

Mooncake tool-and-agent trace, four one-minute windows (1,738 requests), arrivals slowed 15x, KV pool pinned at 160k tokens, Qwen3-8B bf16 on one A100 80GB. Change against `fcfs` is averaged over the windows.

| Row | JCT mean | JCT p50 | JCT p99 | TTFT | TPOT |
| --- | --- | --- | --- | --- | --- |
| fcfs (default) | 17.5 s | 14.0 s | 74.2 s | 8.3 s | 646 ms |
| lpm | -5% | -3% | -1% | -28% | +61% |
| fcfs + prefill-decode-interval 8 | -5% | -8% | +2% | +19% | -90% |
| shortest-prefill-first | -12% | -6% | -17% | -54% | +62% |
| lpm + mixed 8k chunks | -12% | -15% | -1% | -24% | -54% |
| FastServe placement (full prompt) | -24% | -43% | 0% | -28% | -81% |
| sj-mlfq + gate | -30% | -51% | -1% | -40% | -89% |

Throughput is 0.48 requests per second in every row (under capacity). Every row, including the saturation runs, the natural-pool runs, the conversation trace, the repeats and the rows that lost, is in `results/FINAL_TABLES.md`; `LOG.md` is the chronology with each round's numbers and what was concluded from them; `DESIGN.md` explains the mechanism and what was cut.

## Reproduce

- `mooncake_replay.py` replays a window of the public Mooncake traces with their prefix sharing preserved (each 512-token block id becomes a fixed block of token ids).
- `run_mooncake_main.sh <trace> <slowdown> <minutes> <start-minute>` runs one row set; `run_experiment.py` starts the server, replays, and writes `summary.json` and `config.json` per row. `run_review_final.sh` runs the whole table on one pod.
- `analyze_final.py` regenerates the tables; `returning_split.py <toolagent_trace.jsonl>` splits returning turns from fresh prompts; `count_inflight_prefix.py` counts how often re-pricing can change a level.
- `profile_prefill.py` measures the prefill latency grid used to seed the cost model; `prefill_profile_a100-sxm4-80gb.json` is the A100 grid.
- `test_skip_join_mlfq.py` runs the unit checks without a GPU: `python3 cache_aware_mlfq/test_skip_join_mlfq.py`.
- The `pod_*.sh`, `sync_to_pod.sh`, `pull_results.sh` and `idle_stop.sh` scripts manage the RunPod A100 pods the rows ran on.

## Pull requests to SGLang

- [#45688](https://github.com/sgl-project/sglang/pull/45688): skip-join MLFQ with the prefill gate and cache-aware placement.
- [#45686](https://github.com/sgl-project/sglang/pull/45686): priority preemption when KV memory runs out, and an option to keep KV on retraction.
- [#45685](https://github.com/sgl-project/sglang/pull/45685): a chunked prefill continuing into a full KV pool crashed the server.
- [#45687](https://github.com/sgl-project/sglang/pull/45687): the built-in Mooncake dataset loses the trace's prefix sharing.
