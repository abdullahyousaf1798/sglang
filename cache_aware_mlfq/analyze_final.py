"""Tables for the final-design rows (and any Mooncake row with a TAG suffix).

    python cache_aware_mlfq/analyze_final.py [results_dir] [--trace toolagent|conversation] [--tag m1-sd15]

Rows are results/<pool>-<trace>-<config>-s<window>[-<tag>], pool "mooncake" (natural
GPU KV pool) or "mooncake160k" (pinned at 160k tokens). For each (pool, trace, tag) it
prints every config's mean over windows and its change against fcfs (geometric mean of
per-window ratios), the per-window mean JCT, and the short-output / long-output split.
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"(mooncake(?:160k)?)-(toolagent|conversation)-(.+?)-s(\d+)(?:-(m1(?:-sd\d+)?(?:-r2)?))?$")
METRICS = [  # header, summary key, scale, lower is better
    ("JCT mean", "average_latency", 1), ("JCT p50", "p50_latency", 1), ("JCT p99", "p99_latency", 1),
    ("TTFT", "average_ttft", 1), ("TPOT ms", "average_tpot", 1000), ("per-token ms", "average_per_token_latency", 1000),
]
SHORT_OUT, LONG_OUT = 30, 200


def load(results: Path) -> dict:
    runs = defaultdict(dict)  # (pool, trace, tag) -> {(config, window): path}
    for path in sorted(results.iterdir()):
        m = NAME.fullmatch(path.name)
        if m and (path / "summary.json").exists():
            pool, trace, config, window, tag = m.groups()
            runs[(pool, trace, tag or "0.4min")][(config, int(window))] = path
    return runs


def summary(path: Path) -> dict:
    data = json.loads((path / "summary.json").read_text())
    return data["summary"] | data.get("server_counters", {})


def geo(ratios):
    return math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else float("nan")


def split(path: Path):
    reqs = [json.loads(l) for l in (path / "requests.jsonl").read_text().splitlines()]
    reqs = [r for r in reqs if r["error"] is None]
    short = [r for r in reqs if r["output_len"] <= SHORT_OUT]
    long_ = [r for r in reqs if r["output_len"] > LONG_OUT]
    p99 = lambda xs: sorted(xs)[int(0.99 * len(xs))] if xs else float("nan")
    mean = lambda rs, k: statistics.mean(r[k] for r in rs) if rs else float("nan")
    return (mean(short, "e2e_s"), mean(short, "ttft_s"), mean(long_, "e2e_s"), p99([r["e2e_s"] for r in long_]))


def group_table(key, rows: dict) -> None:
    pool, trace, tag = key
    configs = sorted({c for c, _ in rows})
    windows = sorted({w for _, w in rows})
    print(f"\n## {pool} / {trace} / {tag}: windows {windows}")
    print("| config | windows | " + " | ".join(h for h, _, _ in METRICS) + " | thr | preempt | done |")
    print("|---|---|" + "---|" * (len(METRICS) + 3))
    order = sorted(configs, key=lambda c: statistics.mean(summary(rows[(c, w)])["average_latency"] for w in windows if (c, w) in rows))
    for c in order:
        done = [w for w in windows if (c, w) in rows]
        cells = []
        for _, k, scale in METRICS:
            vals = [summary(rows[(c, w)])[k] * scale for w in done]
            base = [summary(rows[("fcfs", w)])[k] * scale for w in done if ("fcfs", w) in rows]
            ratios = [v / b for v, b in zip(vals, base) if b] if len(base) == len(vals) else []
            change = "" if c == "fcfs" or not ratios else f" ({(geo(ratios) - 1) * 100:+.0f}%)"
            cells.append(f"{statistics.mean(vals):.2f}{change}")
        s = [summary(rows[(c, w)]) for w in done]
        thr = statistics.mean(x["throughput"] for x in s)
        pre = statistics.mean(x.get("retracted_requests", 0) for x in s)
        comp = f"{sum(x['completed'] for x in s)}/{sum(x['total_requests'] for x in s)}"
        print(f"| {c} | {len(done)} | " + " | ".join(cells) + f" | {thr:.2f} | {pre:.0f} | {comp} |")
    print("\nMean JCT per window:")
    print("| config | " + " | ".join(f"w{w}" for w in windows) + " |")
    print("|---|" + "---|" * len(windows))
    for c in order:
        print(f"| {c} | " + " | ".join(f"{summary(rows[(c, w)])['average_latency']:.1f}" if (c, w) in rows else "-" for w in windows) + " |")
    print(f"\nShort outputs (<= {SHORT_OUT} tokens) and long outputs (> {LONG_OUT}), mean over windows:")
    print("| config | short JCT | short TTFT | long JCT | long p99 |")
    print("|---|---|---|---|---|")
    for c in order:
        parts = [split(rows[(c, w)]) for w in windows if (c, w) in rows]
        means = [statistics.mean(p[i] for p in parts) for i in range(4)]
        print(f"| {c} | " + " | ".join(f"{m:.1f}" for m in means) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(HERE / "results"))
    ap.add_argument("--trace", default=None)
    ap.add_argument("--tag", default=None, help="e.g. m1-sd15; default: every tag")
    args = ap.parse_args()
    runs = load(Path(args.results))
    for key in sorted(runs):
        pool, trace, tag = key
        if args.trace and trace != args.trace:
            continue
        if args.tag and tag != args.tag:
            continue
        group_table(key, runs[key])


if __name__ == "__main__":
    main()
