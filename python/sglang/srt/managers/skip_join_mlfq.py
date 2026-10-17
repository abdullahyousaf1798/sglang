"""Skip-join multi-level feedback queue (FastServe) with optional cache-aware placement.

A request joins the lowest queue level whose quantum covers its predicted first
iteration (its prefill), instead of always starting at the top. After each
forward pass it is charged the pass's wall time and drops one level once it has
used up its level's quantum; a request left waiting too long is promoted back to
level 0. The level is stored as ``req.priority = -level``, so SGLang's priority
preemption, retraction and eviction all follow MLFQ order.

Placement cost is either FastServe's (a fresh prefill of the whole prompt) or
cache-aware (the uncached suffix prefilled on top of the cached prefix).

``MlfqLoadAhead`` is FastServe's proactive swap-in on top of HiCache: it copies
the CPU-cached prefixes of the next waiting requests back to the GPU before they
are admitted.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.evict_policy import MlfqTierStrategy
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

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

# Load-ahead limits; arbitrary, not tuned. A copy carries at most 32k tokens
# (about 4.5 GB for Qwen3-8B), skips CPU-only prefixes under 256 tokens (cheap to
# load at admission), considers the first 8 waiting requests, and leaves 10% of
# the KV pool free for running requests to grow into.
LOAD_AHEAD_MAX_TOKENS = 32768
LOAD_AHEAD_MIN_TOKENS = 256
LOAD_AHEAD_CANDIDATES = 8
LOAD_AHEAD_HEADROOM = 0.1
LOAD_AHEAD_LOG_INTERVAL = 20


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
        claim_strategy: Optional[MlfqTierStrategy] = None,
        node_of: Optional[Callable[[Any], Any]] = None,
    ):
        """``claim_strategy`` (with ``node_of``, which resolves ``req.last_node`` to a tree
        node) enables two-tier eviction: waiting requests' cached prefixes are claimed
        at their level each step, so unclaimed KV is evicted first."""
        if cost_mode not in COST_MODES:
            raise ValueError(f"--sj-mlfq-cost must be one of {COST_MODES}, got {cost_mode!r}")
        if profile_path is None:
            raise ValueError("--schedule-policy sj-mlfq requires --sj-mlfq-profile")
        if base_quantum_s <= 0:
            raise ValueError(f"--sj-mlfq-base-quantum-ms must be positive, got {base_quantum_s * 1000}")
        if (claim_strategy is None) != (node_of is None):
            raise ValueError("claim_strategy and node_of must be given together")
        self._base_quantum_s = base_quantum_s
        self._claim_strategy = claim_strategy
        self._node_of = node_of
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
        if self._claim_strategy is not None:
            self._claim_waiting_prefixes(waiting_queue)

    def _claim_waiting_prefixes(self, waiting_queue: List[Req]) -> None:
        """Claim each waiting request's cached prefix (from the match this step) at its level."""
        self._claim_strategy.begin_step()
        for req in waiting_queue:
            if req.last_node is None:
                continue
            node = self._node_of(req.last_node)
            while node.parent is not None:  # the root is never evicted
                self._claim_strategy.claim(node, req.priority)
                node = node.parent

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


class MlfqLoadAhead:
    """Proactive load-back for ``--sj-mlfq-load-ahead``.

    FastServe swaps in the KV of the jobs it will run next while other jobs run.
    Here "next" is the MLFQ waiting order (FastServe's estimated next scheduled
    time), and the copy is HiCache's own load-back, started before admission
    instead of at it, so it overlaps other batches instead of stalling the
    admitting prefill. A copy starts only when no other load-back is in flight,
    which keeps it within the cache controller's three layer-counter slots.
    """

    def __init__(self, *, tree_cache: UnifiedRadixCache):
        self._tree_cache = tree_cache
        self._producer = -1  # layer-counter slot of the copy in flight
        self._anchor_ids: List[Any] = []
        self._copies = 0
        self._tokens = 0

    def consumer_index(self) -> int:
        """Layer-counter slot a new prefill batch must wait on, or -1 when no copy is in flight.

        HiCache makes only the batch that started a load wait for it, but a request
        admitted while a copy is in flight may read the KV that copy is writing.
        """
        if self._producer >= 0 and not any(
            anchor in self._tree_cache.ongoing_load_back for anchor in self._anchor_ids
        ):
            self._producer, self._anchor_ids = -1, []
        return self._producer

    def step(self, waiting_queue: List[Req]) -> None:
        """Start one copy for the next waiting requests whose cached prefix is only on the CPU."""
        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
        from sglang.srt.mem_cache.common import match_kv_cache

        tree = self._tree_cache
        if self.consumer_index() >= 0 or tree.cache_controller.ack_load_queue:
            return
        allocator = tree.token_to_kv_pool_allocator
        budget = min(
            LOAD_AHEAD_MAX_TOKENS,
            allocator.available_size()
            + tree.evictable_size()
            - int(LOAD_AHEAD_HEADROOM * allocator.size),
        )
        anchors, tokens = [], 0
        for req in waiting_queue[:LOAD_AHEAD_CANDIDATES]:
            match_kv_cache(req, tree)
            if req.host_hit_length < LOAD_AHEAD_MIN_TOKENS:
                continue
            if tokens + req.host_hit_length > budget:
                break  # keep the copy in waiting order
            loaded, _ = tree.init_load_back(
                InitLoadBackParams(
                    best_match_node=req.best_match_node,
                    host_hit_length=req.host_hit_length,
                    mem_quota=budget - tokens,
                    req=req,
                )
            )
            if loaded > 0:
                anchors.append(req.best_match_node)
                tokens += loaded
        if not anchors:
            return
        self._producer = tree.ready_to_load_host_cache()
        self._anchor_ids = anchors
        self._copies += 1
        self._tokens += tokens
        if self._copies % LOAD_AHEAD_LOG_INTERVAL == 0:
            logger.info(
                "sj-mlfq load-ahead: %d copies, %d tokens", self._copies, self._tokens
            )
