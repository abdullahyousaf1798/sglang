"""Combine benchmark runs into one comparison table (Markdown).

    python cache_aware_mlfq/make_table.py [results_dir] [run ...]

Without run names, every folder in results_dir that has a summary.json is included.
"""

import json
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
    ("Preemptions", lambda s, c: f"{c.get('retracted_requests', 0):.0f}"),
]


def share(counters: dict, mode: str) -> str:
    total = sum(counters.get(f"prefill_tokens_{m}", 0) for m in ("input", "device_hit", "host_hit"))
    return f"{counters.get(f'prefill_tokens_{mode}', 0) / total:.0%}" if total else "-"


def main() -> None:
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
