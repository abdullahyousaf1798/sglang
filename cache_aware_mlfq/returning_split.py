"""Returning turns versus fresh prompts for final-table rows.

    python cache_aware_mlfq/returning_split.py <toolagent_trace.jsonl> [--tag m1-sd15] [--pool mooncake160k]

A returning turn has at least half its prompt in blocks an earlier request of the
window sent (analyze_mooncake.returning_mask). Prints mean JCT and TTFT per group.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_mooncake as am  # noqa: E402

CONFIGS = ["fcfs", "lpm", "shortest-prefill-first", "fcfs-interval8", "lpm-mixed8k",
           "fastserve-final", "ours-final", "ours-final-yield"]


def split(results: Path, trace: str, pool: str, cfg: str, tag: str, windows, minutes: float):
    ret, fresh = [], []
    for w in windows:
        d = results / f"{pool}-toolagent-{cfg}-s{w}-{tag}"
        if not (d / "requests.jsonl").exists():
            return None
        mask = am.returning_mask(trace, w, minutes)
        for r in (json.loads(l) for l in open(d / "requests.jsonl")):
            if r["error"] is None:
                (ret if mask[r["index"]] else fresh).append(r)
    m = lambda xs, k: statistics.mean(x[k] for x in xs)
    return len(ret), len(fresh), m(ret, "e2e_s"), m(ret, "ttft_s"), m(fresh, "e2e_s"), m(fresh, "ttft_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--tag", default="m1-sd15")
    ap.add_argument("--pool", default="mooncake160k", choices=["mooncake160k", "mooncake"])
    ap.add_argument("--windows", default="10,25,40,55")
    ap.add_argument("--minutes", type=float, default=1.0)
    a = ap.parse_args()
    windows = [int(w) for w in a.windows.split(",")]
    print(f"## {a.pool} / {a.tag}: returning turns vs fresh prompts (mean over windows {windows})")
    print("| config | n ret / fresh | ret JCT | ret TTFT | fresh JCT | fresh TTFT |")
    print("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        v = split(Path(a.results), a.trace, a.pool, cfg, a.tag, windows, a.minutes)
        if v:
            print(f"| {cfg} | {v[0]} / {v[1]} | {v[2]:.1f} | {v[3]:.1f} | {v[4]:.1f} | {v[5]:.1f} |")


if __name__ == "__main__":
    main()
