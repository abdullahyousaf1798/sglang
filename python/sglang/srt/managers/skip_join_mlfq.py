"""Skip-join multi-level feedback queue (FastServe) with optional cache-aware placement.

A request joins the lowest queue level whose quantum covers its predicted first
iteration (its prefill), instead of always starting at the top. After each
forward pass it is charged the pass's wall time and drops one level once it has
used up its level's quantum; a request left waiting too long is promoted back to
level 0. The level is stored as ``req.priority = -level``, so SGLang's priority
preemption, retraction and eviction all follow MLFQ order.

Placement cost is either FastServe's (a fresh prefill of the whole prompt) or
cache-aware (the uncached suffix prefilled on top of the cached prefix).
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import time
from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# FastServe's settings (Wu et al., arXiv:2305.05920): a 10 ms base quantum that
# doubles per level, and promotion after 3 s without service, checked every
# 1000 scheduling steps. The base quantum is tunable because it should track the
# hardware's iteration time; FastServe's 10 ms is the default.
DEFAULT_BASE_QUANTUM_S = 0.010
QUANTUM_MULTIPLIER = 2.0
STARVATION_S = 3.0
STARVATION_CHECK_INTERVAL = 1000

COST_MODES = ("prompt", "cache-aware")


def _interpolate(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation that extends the end segments past the grid."""
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class PrefillCostModel:
    """Predicted prefill time from a profiled (uncached, cached) latency grid.

    The profile comes from ``cache_aware_mlfq/profile_prefill.py``; its latencies
    include a fixed per-request overhead, which is subtracted to estimate the
    forward pass alone.
    """

    def __init__(self, profile_path: str):
        with open(profile_path) as f:
            profile = json.load(f)
        self._overhead_s = profile["request_overhead_ms"] / 1000
        rows: dict = {}
        for cell in profile["cells"]:
            rows.setdefault(cell["cached"], []).append(
                (cell["uncached"], cell["latency_ms"] / 1000)
            )
        self._cached_grid = sorted(rows)
        self._rows: List[Tuple[List[int], List[float]]] = []
        for cached in self._cached_grid:
            points = sorted(rows[cached])
            self._rows.append(([u for u, _ in points], [s for _, s in points]))

    def predict_s(self, *, uncached: int, cached: int) -> float:
        at_uncached = [_interpolate(us, ss, uncached) for us, ss in self._rows]
        latency = _interpolate(self._cached_grid, at_uncached, cached)
        return max(0.0, latency - self._overhead_s)


class SkipJoinMlfq:
    """Queue levels, demotion and starvation promotion for ``--schedule-policy sj-mlfq``."""

    def __init__(
        self,
        *,
        cost_mode: str,
        profile_path: Optional[str],
        base_quantum_s: float = DEFAULT_BASE_QUANTUM_S,
    ):
        if cost_mode not in COST_MODES:
            raise ValueError(f"--sj-mlfq-cost must be one of {COST_MODES}, got {cost_mode!r}")
        if profile_path is None:
            raise ValueError("--schedule-policy sj-mlfq requires --sj-mlfq-profile")
        if base_quantum_s <= 0:
            raise ValueError(f"--sj-mlfq-base-quantum-ms must be positive, got {base_quantum_s * 1000}")
        self._base_quantum_s = base_quantum_s
        self._cache_aware = cost_mode == "cache-aware"
        self._cost_model = PrefillCostModel(profile_path)
        self._steps = 0
        self._placed_per_level: Counter = Counter()
        self._demotions = 0
        self._promotions = 0

    def order(self, waiting_queue: List[Req]) -> None:
        """Place new requests, promote starved ones, then sort by (level, level entry time)."""
        now = time.monotonic()
        for req in waiting_queue:
            if req.mlfq_level is None:
                self._place(req, now)
        self._steps += 1
        if self._steps % STARVATION_CHECK_INTERVAL == 0:
            for req in waiting_queue:
                if req.mlfq_level > 0 and now - req.mlfq_last_service_time >= STARVATION_S:
                    # Keeps its entry time, so it lands ahead of newer level-0 requests.
                    self._set_level(req, 0)
                    self._promotions += 1
            logger.info(
                "sj-mlfq: placed per level %s, demotions %d, starvation promotions %d",
                dict(sorted(self._placed_per_level.items())),
                self._demotions,
                self._promotions,
            )
        waiting_queue.sort(key=lambda r: (r.mlfq_level, r.mlfq_level_entry_time))

    def charge(self, reqs: List[Req], elapsed_s: float) -> None:
        """Charge one forward pass to each request in it; demote those past their quantum."""
        now = time.monotonic()
        for req in reqs:
            if req.mlfq_level is None:
                continue
            req.mlfq_service_s += elapsed_s
            req.mlfq_last_service_time = now
            if req.mlfq_service_s > self.quantum_s(req.mlfq_level):
                self._set_level(req, req.mlfq_level + 1)
                req.mlfq_level_entry_time = now
                self._demotions += 1

    def quantum_s(self, level: int) -> float:
        return self._base_quantum_s * QUANTUM_MULTIPLIER**level

    def _place(self, req: Req, now: float) -> None:
        prompt_len = len(req.origin_input_ids)
        if self._cache_aware:
            # Counts both GPU-resident and CPU-resident (HiCache) prefix hits.
            cached = req.num_matched_prefix_tokens
            cost_s = self._cost_model.predict_s(uncached=prompt_len - cached, cached=cached)
        else:
            cost_s = self._cost_model.predict_s(uncached=prompt_len, cached=0)
        level = 0
        if cost_s > self._base_quantum_s:
            level = math.ceil(math.log(cost_s / self._base_quantum_s, QUANTUM_MULTIPLIER))
        self._set_level(req, level)
        req.mlfq_level_entry_time = now
        req.mlfq_last_service_time = now
        self._placed_per_level[level] += 1

    @staticmethod
    def _set_level(req: Req, level: int) -> None:
        req.mlfq_level = level
        req.priority = -level
        req.mlfq_service_s = 0.0
