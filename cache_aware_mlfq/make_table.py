"""Combine benchmark runs into one comparison table (Markdown).

    python cache_aware_mlfq/make_table.py [results_dir] [run ...]
    python cache_aware_mlfq/make_table.py --mean [results_dir] [prefix]

Without run names, every folder in results_dir that has a summary.json is included.
With --mean, runs named <config>-s<seed> are grouped by config and shown as
mean ± standard deviation over seeds; prefix (e.g. sharegpt-) selects the runs.
"""

import json
import re
import statistics
import sys
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"

COLUMNS = [
    # header, function of (summary, counters) -> formatted cell
    ("Job completion, mean (s)", lambda s, c: f"{s['average_latency']:.2f}"),
    ("Job completion, p99 (s)", lambda s, c: f"{s['p99_latency']:.2f}"),
    ("TTFT, mean (s)", lambda s, c: f"{s['average_ttft']:.2f}"),
    ("TTFT, p99 (s)", lambda s, c: f"{s['p99_ttft']:.2f}"),
    ("TPOT, mean (ms)", lambda s, c: f"{s['average_tpot'] * 1000:.1f}"),
    ("Throughput (req/s)", lambda s, c: f"{s['throughput']:.2f}"),
    ("Prompt tokens from GPU cache", lambda s, c: share(c, "device_hit")),
    ("Prompt tokens from CPU cache", lambda s, c: share(c, "host_hit")),
    ("Preempted or retracted", lambda s, c: f"{c.get('retracted_requests', 0):.0f}"),
    ("Decoded tokens at preemption", lambda s, c: f"{c.get('retracted_output_tokens', 0):.0f}"),
    ("Prefill tokens recomputed after preemption", lambda s, c: recomputed(c)),
]


def recomputed(counters: dict) -> str:
    if "prefill_compute_tokens_all" not in counters:
        return "-"  # recorded from step 4a on
    return f"{counters['prefill_compute_tokens_all'] - counters.get('prefill_tokens_input', 0):.0f}"


def share(counters: dict, mode: str) -> str:
    total = sum(counters.get(f"prefill_tokens_{m}", 0) for m in ("input", "device_hit", "host_hit"))
    return f"{counters.get(f'prefill_tokens_{mode}', 0) / total:.0%}" if total else "-"


MEAN_COLUMNS = [
    # header, function of (summary, counters) -> number
    ("Job completion, mean (s)", lambda s, c: s["average_latency"]),
    ("Job completion, p99 (s)", lambda s, c: s["p99_latency"]),
    ("TTFT, mean (s)", lambda s, c: s["average_ttft"]),
    ("TTFT, p99 (s)", lambda s, c: s["p99_ttft"]),
    ("TPOT, mean (ms)", lambda s, c: s["average_tpot"] * 1000),
    ("Throughput (req/s)", lambda s, c: s["throughput"]),
    ("Prompt tokens from CPU cache (%)", lambda s, c: 100 * fraction(c, "host_hit")),
    ("Preempted or retracted", lambda s, c: c.get("retracted_requests", 0)),
    ("Load-back copy time, total (s)", lambda s, c: c.get("load_back_seconds", 0)),
]


# Mooncake runs (mooncake_replay.py) also report the median and FastServe's metric,
# end-to-end latency divided by output length.
MOONCAKE_MEAN_COLUMNS = [
    ("Job completion, mean (s)", lambda s, c: s["average_latency"]),
    ("Job completion, p50 (s)", lambda s, c: s["p50_latency"]),
    ("Job completion, p99 (s)", lambda s, c: s["p99_latency"]),
    ("TTFT, mean (s)", lambda s, c: s["average_ttft"]),
    ("TTFT, p99 (s)", lambda s, c: s["p99_ttft"]),
    ("TPOT, mean (ms)", lambda s, c: s["average_tpot"] * 1000),
    ("Per-token latency, mean (ms)", lambda s, c: s["average_per_token_latency"] * 1000),
    ("Throughput (req/s)", lambda s, c: s["throughput"]),
    ("Preempted or retracted", lambda s, c: c.get("retracted_requests", 0)),
    ("Load-back copy time, total (s)", lambda s, c: c.get("load_back_seconds", 0)),
    ("Loaded ahead (k tokens)", lambda s, c: c.get("load_ahead_tokens", 0) / 1000),
]


def fraction(counters: dict, mode: str) -> float:
    total = sum(counters.get(f"prefill_tokens_{m}", 0) for m in ("input", "device_hit", "host_hit"))
    return counters.get(f"prefill_tokens_{mode}", 0) / total if total else 0.0


def mean_table(results: Path, prefix: str) -> None:
    groups: dict = {}
    for path in sorted(results.iterdir()):
        match = re.fullmatch(rf"{re.escape(prefix)}(.+)-s(\d+)", path.name)
        if match and (path / "summary.json").exists():
            groups.setdefault(match.group(1), []).append(json.loads((path / "summary.json").read_text()))
    columns = MOONCAKE_MEAN_COLUMNS if prefix.startswith("mooncake") else MEAN_COLUMNS
    print("| Config | Seeds | " + " | ".join(h for h, _ in columns) + " |")
    print("| --- | --- |" + " --- |" * len(columns))
    for config, runs in groups.items():
        cells = []
        for _, metric in columns:
            values = [metric(r["summary"], r.get("server_counters", {})) for r in runs]
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):.2f} ± {spread:.2f}")
        print(f"| {config} | {len(runs)} | " + " | ".join(cells) + " |")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--mean":
        results = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RESULTS
        mean_table(results, sys.argv[3] if len(sys.argv) > 3 else "")
        return
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    names = sys.argv[2:] or sorted(p.name for p in results.iterdir() if (p / "summary.json").exists())
    print("| Run | " + " | ".join(h for h, _ in COLUMNS) + " |")
    print("| --- |" + " --- |" * len(COLUMNS))
    for name in names:
        data = json.loads((results / name / "summary.json").read_text())
        summary, counters = data["summary"], data.get("server_counters", {})
        print(f"| {name} | " + " | ".join(cell(summary, counters) for _, cell in COLUMNS) + " |")


if __name__ == "__main__":
    main()
