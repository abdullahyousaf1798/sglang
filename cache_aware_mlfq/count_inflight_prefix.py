"""How often would re-pricing change a request's MLFQ level?

For each request in a trace window (empty cache at the window start), the leading
blocks already sent by an earlier request of the window are its shared prefix. Blocks
first sent less than DELTA seconds of trace time before this request's arrival are
taken as "in flight": not yet cached when the request is placed, cached by the time it
is admitted. The level at placement prices the full suffix beyond the cached-at-
placement prefix; the true level prices only the suffix beyond the whole shared
prefix. A request counts when the two levels differ.

    python3 -I count_inflight_prefix.py <trace.jsonl> --windows 10 25 40 55 --minutes 1 --deltas 0.3 1.0
"""
import argparse
import json
import math

BLOCK = 512
SEED = (7.2e-5, 2.8e-9, 6e-7)  # OnlinePrefillCostModel seed: per uncached, per u*c, per cached
BASE_Q = 0.010


def cost_s(uncached: int, cached: int) -> float:
    a, b, d = SEED
    return a * uncached + b * uncached * cached + d * cached


def level(uncached: int, cached: int) -> int:
    c = cost_s(uncached, cached)
    return 0 if c <= BASE_Q else math.ceil(math.log(c / BASE_Q, 2))


def load_window(path, start_min, minutes):
    begin, end = start_min * 60_000, (start_min + minutes) * 60_000
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if begin <= r["timestamp"] < end:
                out.append(r)
    out.sort(key=lambda r: r["timestamp"])
    return out


def analyze(window, delta_s):
    first_sent = {}  # block id -> trace time (s) of the first request that sent it
    n = changed = inflight_any = returning_changed = 0
    level_gap = []
    for r in window:
        t = r["timestamp"] / 1000.0
        blocks, L = r["hash_ids"], r["input_length"]
        k_all = 0
        for b in blocks:
            if b not in first_sent:
                break
            k_all += 1
        # k_old counts leading blocks whose first sender is at least delta_s older,
        # stopping at the first in-flight block.
        k_old = 0
        for b in blocks:
            if b not in first_sent or first_sent[b] > t - delta_s:
                break
            k_old += 1
        cached_all, cached_old = min(k_all * BLOCK, L), min(k_old * BLOCK, L)
        lp = level(L - cached_old, cached_old)
        lt = level(L - cached_all, cached_all)
        n += 1
        if k_all > k_old:
            inflight_any += 1
        if lt < lp:
            changed += 1
            level_gap.append(lp - lt)
            if cached_all >= 0.5 * L:
                returning_changed += 1
        for b in blocks:
            first_sent.setdefault(b, t)
    return n, inflight_any, changed, returning_changed, (sum(level_gap) / len(level_gap) if level_gap else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--windows", type=float, nargs="+", default=[10, 25, 40, 55])
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.3, 1.0])
    a = ap.parse_args()
    print("| window | delta (trace s / wall s at x15) | requests | any in-flight shared prefix | level would change | of which returning turns | mean levels gained |")
    print("|---|---|---|---|---|---|---|")
    for w in a.windows:
        win = load_window(a.trace, w, a.minutes)
        for d in a.deltas:
            n, inflight, changed, ret, gap = analyze(win, d)
            print(f"| {w:g} | {d:g} / {d*15:g} | {n} | {inflight} ({100*inflight/n:.0f}%) | {changed} ({100*changed/n:.0f}%) | {ret} | {gap:.1f} |")


if __name__ == "__main__":
    main()
