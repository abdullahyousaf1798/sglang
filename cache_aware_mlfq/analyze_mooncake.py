"""Tables for the Mooncake rows: per-window results, change against fcfs, the 2x2
placement x eviction effects, and returning turns versus fresh prompts.

    python cache_aware_mlfq/analyze_mooncake.py <trace.jsonl> [results_dir]

Rows are results/<prefix>-toolagent-<config>-s<window>; prefix "mooncake" is the
natural GPU KV pool and "mooncake160k" the pool pinned at 160k tokens. The trace
file is the one replayed, used to tell returning turns from fresh prompts.
"""

import json
import math
import re
import statistics
import sys
from pathlib import Path

from mooncake_replay import BLOCK_TOKENS, load_window

HERE = Path(__file__).resolve().parent
REGIMES = {"mooncake": "natural pool (393k tokens)", "mooncake160k": "pool pinned at 160k tokens"}
METRICS = [  # header, summary key, scale
    ("JCT mean (s)", "average_latency", 1),
    ("JCT p50 (s)", "p50_latency", 1),
    ("JCT p90 (s)", "p90_latency", 1),
    ("JCT p99 (s)", "p99_latency", 1),
    ("TTFT mean (s)", "average_ttft", 1),
    ("TPOT mean (ms)", "average_tpot", 1000),
    ("Per-token latency (ms)", "average_per_token_latency", 1000),
    ("Throughput (req/s)", "throughput", 1),
]
# Returning turn: at least half its prompt is a prefix an earlier request in the window sent.
RETURNING_SHARE = 0.5


def load_runs(results: Path) -> dict:
    runs = {}
    for path in results.iterdir():
        m = re.fullmatch(r"(mooncake(?:160k)?)-toolagent-(.+)-s(\d+)", path.name)
        if m and (path / "summary.json").exists():
            runs[(m.group(1), m.group(2), int(m.group(3)))] = path
    return runs


def summary(path: Path) -> dict:
    data = json.loads((path / "summary.json").read_text())
    return data["summary"] | data["server_counters"]


def geo_mean_change(ratios: list) -> str:
    """Mean change across windows as a geometric mean of per-window ratios."""
    if not ratios:
        return "-"
    return f"{(math.exp(statistics.mean(math.log(r) for r in ratios)) - 1) * 100:+.0f}%"


def regime_table(runs: dict, regime: str) -> None:
    windows = sorted({w for (r, _, w) in runs if r == regime})
    configs = sorted({c for (r, c, _) in runs if r == regime})
    print(f"\n## {REGIMES[regime]}: windows {windows}\n")
    print("| Config | Windows | " + " | ".join(h for h, _, _ in METRICS)
          + " | Preempted | Reloaded from CPU (k tok) | Loaded ahead (k tok) |")
    print("| --- | --- |" + " --- |" * (len(METRICS) + 3))
    for config in configs:
        done = [w for w in windows if (regime, config, w) in runs]
        rows = [summary(runs[(regime, config, w)]) for w in done]
        cells = []
        for _, key, scale in METRICS:
            mean = statistics.mean(r[key] * scale for r in rows)
            base = [summary(runs[(regime, "fcfs", w)])[key] for w in done
                    if (regime, "fcfs", w) in runs]
            ratios = [r[key] / b for r, b in zip(rows, base) if b] if len(base) == len(rows) else []
            change = "" if config == "fcfs" else f" ({geo_mean_change(ratios)})"
            cells.append(f"{mean:.2f}{change}")
        extra = [statistics.mean(r.get(k, 0) for r in rows) / d
                 for k, d in (("retracted_requests", 1), ("load_back_tokens", 1000),
                              ("load_ahead_tokens", 1000))]
        print(f"| {config} | {len(done)} | " + " | ".join(cells)
              + f" | {extra[0]:.0f} | {extra[1]:.0f} | {extra[2]:.0f} |")


def effect(runs: dict, regime: str, base: str, variant: str, key: str = "average_latency") -> str:
    ratios = [summary(runs[(regime, variant, w)])[key] / summary(runs[(regime, base, w)])[key]
              for (r, c, w) in runs if r == regime and c == base and (regime, variant, w) in runs]
    return f"{geo_mean_change(ratios)} over {len(ratios)} windows"


def two_by_two(runs: dict, regime: str) -> None:
    print(f"\n## 2x2 effects, {REGIMES[regime]} (change in mean JCT; negative is better)\n")
    for gate in ("", "-gate"):
        lru, tier = f"fastserve-lru{gate}", f"fastserve{gate}"
        ours_lru = "ours-lru-eviction" if not gate else "ours-lru-gate"
        ours = f"ours{gate}"
        label = "with the prefill gate" if gate else "without the gate"
        print(f"- {label}:")
        print(f"  - cache-aware placement, LRU eviction: {effect(runs, regime, lru, ours_lru)}")
        print(f"  - cache-aware placement, two-tier eviction: {effect(runs, regime, tier, ours)}")
        print(f"  - two-tier eviction, FastServe placement: {effect(runs, regime, lru, tier)}")
        print(f"  - two-tier eviction, cache-aware placement: {effect(runs, regime, ours_lru, ours)}")
        print(f"  - both, against FastServe ported: {effect(runs, regime, lru, ours)}")


def returning_mask(trace: str, window: int, minutes: float) -> list:
    seen, mask = set(), []
    for record in load_window(trace, window, minutes):
        k = 0
        for block in record["hash_ids"]:
            if block not in seen:
                break
            k += 1
        mask.append(k * BLOCK_TOKENS >= RETURNING_SHARE * record["input_length"])
        seen.update(record["hash_ids"])
    return mask


def split_table(runs: dict, regime: str, trace: str, configs: list) -> None:
    print(f"\n## Returning turns vs fresh prompts, {REGIMES[regime]} (mean over windows)\n")
    print("| Config | Returning: JCT mean (s) | Returning: TTFT mean (s) | Fresh: JCT mean (s) | Fresh: TTFT mean (s) |")
    print("| --- | --- | --- | --- | --- |")
    for config in configs:
        per_window = []
        for (r, c, w), path in sorted(runs.items()):
            if r != regime or c != config:
                continue
            trace_cfg = json.loads((path / "bench.jsonl").read_text().splitlines()[-1])["trace"]
            mask = returning_mask(trace, trace_cfg["start_min"], trace_cfg["minutes"])
            reqs = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
            groups = {True: [], False: []}
            for req in reqs:
                if req["error"] is None:
                    groups[mask[req["index"]]].append(req)
            per_window.append({g: (statistics.mean(q["e2e_s"] for q in rs), statistics.mean(q["ttft_s"] for q in rs))
                               for g, rs in groups.items() if rs})
        if not per_window:
            continue
        cell = lambda g, i: f"{statistics.mean(p[g][i] for p in per_window if g in p):.2f}"
        print(f"| {config} | {cell(True, 0)} | {cell(True, 1)} | {cell(False, 0)} | {cell(False, 1)} |")


def main() -> None:
    trace = sys.argv[1]
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results"
    runs = load_runs(results)
    for regime in REGIMES:
        if any(r == regime for (r, _, _) in runs):
            regime_table(runs, regime)
    two_by_two(runs, "mooncake160k")
    split_table(runs, "mooncake160k", trace,
                ["fcfs", "fastserve-lru-gate", "ours-lru-gate", "fastserve-gate", "ours-gate"])
    split_table(runs, "mooncake", trace, ["fcfs", "fastserve-gate", "ours-gate"])


if __name__ == "__main__":
    main()
