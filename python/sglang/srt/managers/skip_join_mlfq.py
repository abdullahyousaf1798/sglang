"""Skip-join multi-level feedback queue (FastServe) with optional cache-aware placement.

A request joins the lowest queue level whose quantum covers its predicted first
iteration (its prefill), instead of always starting at the top. While it waits and
has not run yet, it is re-priced every ordering against the current cache and may
only move to a better level, so a request whose shared prefix another request has
just prefilled is not stuck at the level it was given before. After each
forward pass it is charged the pass's wall time and drops one level once it has
used up its level's quantum; a request left waiting too long is promoted back to
level 0. The level is stored as ``req.priority = -level``, so SGLang's priority
preemption, retraction and eviction all follow MLFQ order.

Every attention-parallel rank runs its own scheduler and must place, demote and
promote identically, so the policy never reads the wall clock: ``charge`` receives
each batch's elapsed time and timestamp as measured on one rank and broadcast to
the others (``batch_result_processor._charge_mlfq``), and keeps that timestamp as
its clock for level entry times and the starvation check.

Placement cost is either FastServe's (a fresh prefill of the whole prompt) or
cache-aware (the uncached suffix prefilled on top of the cached prefix). The cost
comes from a profiled latency grid when one is given, otherwise from a seed cost
shape whose compute and attention scales are fitted online to the prefill batches
the scheduler actually runs.

``mlfq_chunk_limit`` is shortest-prefill-first's chunk-sharing rule at MLFQ levels:
a chunked prefill in progress leaves room in its batch for more urgent arrivals.

``MlfqLoadAhead`` is FastServe's proactive swap-in on top of HiCache: it copies
the CPU-cached prefixes of the next waiting requests back to the GPU before they
are admitted.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
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

# Seed for the online cost model, in seconds: per uncached token, per (uncached x
# cached) token pair, per cached token. Fitted to Qwen3-8B on an A100; levels are
# logarithmic, so a factor-two error elsewhere costs one level until the fit corrects.
DEFAULT_COST_SEED = (7.2e-5, 2.8e-9, 6e-7)

# Load-ahead limits; arbitrary, not tuned. A copy carries at most 32k tokens
# (about 4.5 GB for Qwen3-8B), skips CPU-only prefixes under 256 tokens (cheap to
# load at admission), considers the first 8 waiting requests, and leaves 10% of
# the KV pool free for running requests to grow into.
LOAD_AHEAD_MAX_TOKENS = 32768
LOAD_AHEAD_MIN_TOKENS = 256
LOAD_AHEAD_CANDIDATES = 8
LOAD_AHEAD_HEADROOM = 0.1
LOAD_AHEAD_LOG_INTERVAL = 1


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


class OnlinePrefillCostModel:
    """Prefill time as compute * a*u + attention * (b*u*c + d*c), fitted online.

    (a, b, d) is a seed shape: dense compute per uncached token u, attention of new
    tokens over the cached prefix c, and the per-chunk read of the prefix KV. The two
    scale factors are fitted by ridge least squares with exponential forgetting from
    the prefill batches the scheduler runs, so no hardware profile is needed. Two
    factors, not three free coefficients, because u*c and c are nearly collinear in
    real batches and a free fit drifts; a wrong shape costs at most about one level.
    """

    FORGETTING = 0.98
    RIDGE = 0.2  # weight of the seed factors (1, 1): about one typical batch (x^2 ~ 0.1-0.4 s^2)
    FACTOR_RANGE = (0.1, 10.0)

    def __init__(self, seed: Tuple[float, float, float] = DEFAULT_COST_SEED):
        self._seed = seed
        self._factors = [1.0, 1.0]
        self._ata = [[self.RIDGE, 0.0], [0.0, self.RIDGE]]
        self._atb = [self.RIDGE, self.RIDGE]
        self.observations = 0

    def _features(self, rows: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
        a, b, d = self._seed
        compute = attention = 0.0
        for u, c in rows:
            compute += a * u
            attention += b * u * c + d * c
        return compute, attention

    def observe(self, rows: Sequence[Tuple[int, int]], elapsed_s: float) -> None:
        """One forward pass over ``rows`` = [(uncached, cached), ...] that took ``elapsed_s``."""
        if not rows or not 0 < elapsed_s < 60:
            return
        x = self._features(rows)
        lam = self.FORGETTING
        for i in range(2):
            for j in range(2):
                self._ata[i][j] = lam * self._ata[i][j] + x[i] * x[j]
            # The seed anchor is topped up so forgetting never erodes it.
            self._ata[i][i] += (1 - lam) * self.RIDGE
            self._atb[i] = lam * self._atb[i] + x[i] * elapsed_s + (1 - lam) * self.RIDGE
        self.observations += 1
        (p, q), (r, t) = self._ata
        det = p * t - q * r
        if abs(det) < 1e-12:
            return
        lo, hi = self.FACTOR_RANGE
        f0 = (t * self._atb[0] - q * self._atb[1]) / det
        f1 = (p * self._atb[1] - r * self._atb[0]) / det
        self._factors = [min(max(f0, lo), hi), min(max(f1, lo), hi)]

    def predict_s(self, *, uncached: int, cached: int) -> float:
        compute, attention = self._features([(uncached, cached)])
        return max(0.0, self._factors[0] * compute + self._factors[1] * attention)


def mlfq_chunk_limit(
    *,
    chunked_level: Optional[int],
    waiting: Sequence[Tuple[Optional[int], int]],
    budget: int,
    page_size: int,
    gate_level: Optional[int],
) -> Optional[int]:
    """Chunk-token cap for the prefill in progress, or None to keep the full chunk.

    ``waiting`` is [(level, uncached work)] in queue order. Room is reserved for
    waiting requests in a more urgent queue than the chunked request that also pass
    the gate, until the chunk would stop making progress; this is
    shortest-prefill-first's reservation rule with the MLFQ level as urgency.
    """
    if budget < 2 * page_size or chunked_level is None:
        return None
    reserved = 0
    for level, work in waiting:
        if level is None or level >= chunked_level:
            break
        if gate_level is not None and level > gate_level:
            break
        charge = -(-work // page_size) * page_size
        if reserved + charge > budget - page_size:
            break
        reserved += charge
    if not reserved:
        return None
    return (budget - reserved) // page_size * page_size


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
        if base_quantum_s <= 0:
            raise ValueError(f"--sj-mlfq-base-quantum-ms must be positive, got {base_quantum_s * 1000}")
        if (claim_strategy is None) != (node_of is None):
            raise ValueError("claim_strategy and node_of must be given together")
        self._base_quantum_s = base_quantum_s
        self._claim_strategy = claim_strategy
        self._node_of = node_of
        self._cache_aware = cost_mode == "cache-aware"
        # A profile fixes the cost model; without one it is fitted online.
        self._online_model = OnlinePrefillCostModel() if profile_path is None else None
        self._cost_model = self._online_model or PrefillCostModel(profile_path)
        self._steps = 0
        # Timestamp of the last charged batch result, the same on every rank.
        self._clock = 0.0
        self._placed_per_level: Counter = Counter()
        self._demotions = 0
        self._promotions = 0
        self._repricings = 0

    def order(
        self,
        waiting_queue: List[Req],
        temporary_deprioritized: Optional[Set[str]] = None,
    ) -> None:
        """Place new requests, promote starved ones, then sort by (level, level entry time).

        ``temporary_deprioritized`` is SGLang's in-batch prefix-caching set: waiting
        requests whose uncached prefix another waiting request is about to prefill.
        They go to the back for this ordering so the prefix is computed once.
        """
        now = self._clock
        for req in waiting_queue:
            if req.mlfq_level is None:
                self._place(req, now)
            elif req.mlfq_runs == 0:
                self._reprice(req)
        self._steps += 1
        if self._steps % STARVATION_CHECK_INTERVAL == 0:
            for req in waiting_queue:
                if req.mlfq_level > 0 and now - req.mlfq_last_service_time >= STARVATION_S:
                    # Keeps its entry time, so it lands ahead of newer level-0 requests.
                    self._set_level(req, 0)
                    self._promotions += 1
            logger.info(
                "sj-mlfq: placed per level %s, demotions %d, starvation promotions %d, "
                "repricings %d",
                dict(sorted(self._placed_per_level.items())),
                self._demotions,
                self._promotions,
                self._repricings,
            )
        deprioritized = temporary_deprioritized or ()
        waiting_queue.sort(
            key=lambda r: (r.rid in deprioritized, r.mlfq_level, r.mlfq_level_entry_time)
        )
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

    def charge(self, reqs: List[Req], elapsed_s: float, *, now: float) -> None:
        """Charge one forward pass to each request in it; demote those past their quantum.

        ``elapsed_s`` and ``now`` (the pass's end timestamp) must be the same on every
        rank; they advance the policy's clock.
        """
        self._clock = now = max(self._clock, now)
        for req in reqs:
            if req.mlfq_level is None:
                continue
            req.mlfq_runs += 1
            req.mlfq_service_s += elapsed_s
            req.mlfq_last_service_time = now
            if req.mlfq_service_s > self.quantum_s(req.mlfq_level):
                self._set_level(req, req.mlfq_level + 1)
                req.mlfq_level_entry_time = now
                self._demotions += 1

    def observe_batch(self, batch: ScheduleBatch, elapsed_s: float) -> None:
        """Feed a finished prefill (or mixed) batch's rows and wall time to the online fit."""
        if (
            self._online_model is not None
            and batch.forward_mode.is_extend()
            and batch.extend_lens is not None
        ):
            self._online_model.observe(
                list(zip(batch.extend_lens, batch.prefix_lens)), elapsed_s
            )

    @staticmethod
    def prefill_gate_level(running_reqs: List[Req]) -> Optional[int]:
        """Prefill gate: the worst level a waiting request may have to be prefilled now, or None.

        SGLang runs a waiting request's prefill before the running batch's next
        decode step, so a long, low-priority prefill stalls every running request;
        FastServe runs the highest-priority work each iteration instead. Only
        requests at least as urgent as the most urgent running request pass;
        with nothing running, everything does.
        """
        levels = [r.mlfq_level for r in running_reqs if r.mlfq_level is not None]
        return min(levels) if levels else None

    @staticmethod
    def pause_chunk(chunked_level: Optional[int], gate_level: Optional[int]) -> bool:
        """Chunk pause: whether a chunked prefill in progress waits for the gate.

        The gate decides whether a prefill starts; without the pause a started
        chunk runs every step until it is done, so a long, low-priority prompt
        still stalls more urgent decodes once its first chunk has gone in. With
        the pause the continuation obeys the same rule: the chunk waits while a
        running request is more urgent. Its finished chunks stay locked in the
        cache between steps, so waiting costs nothing but the wait. Running
        requests are charged every decode step and demoted past their quanta,
        and no new prefill starts while the chunk waits, so the gate reopens
        within about the chunk's own quantum (the quanta below it sum to less).
        """
        return (
            chunked_level is not None
            and gate_level is not None
            and chunked_level > gate_level
        )

    def quantum_s(self, level: int) -> float:
        return self._base_quantum_s * QUANTUM_MULTIPLIER**level

    def _placement_level(self, req: Req) -> int:
        prompt_len = len(req.origin_input_ids)
        if self._cache_aware:
            # Counts both GPU-resident and CPU-resident (HiCache) prefix hits.
            cached = req.num_matched_prefix_tokens
            cost_s = self._cost_model.predict_s(uncached=prompt_len - cached, cached=cached)
        else:
            cost_s = self._cost_model.predict_s(uncached=prompt_len, cached=0)
        if cost_s <= self._base_quantum_s:
            return 0
        return math.ceil(math.log(cost_s / self._base_quantum_s, QUANTUM_MULTIPLIER))

    def _place(self, req: Req, now: float) -> None:
        level = self._placement_level(req)
        self._set_level(req, level)
        req.mlfq_level_entry_time = now
        req.mlfq_last_service_time = now
        self._placed_per_level[level] += 1

    def _reprice(self, req: Req) -> None:
        # Only toward a better level: a promoted request keeps level 0, and the cost
        # never rises while HiCache counts host-resident prefixes as cached. The level
        # entry time stays, so the request keeps its place among its new peers.
        level = self._placement_level(req)
        if level < req.mlfq_level:
            self._set_level(req, level)
            self._repricings += 1

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
