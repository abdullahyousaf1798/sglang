"""Unit checks for managers/skip_join_mlfq.py; no GPU or SGLang install needed.

    python3 cache_aware_mlfq/test_skip_join_mlfq.py
"""

import importlib.util
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "prefill_profile_4090.json"
_spec = importlib.util.spec_from_file_location(
    "skip_join_mlfq", HERE.parent / "python/sglang/srt/managers/skip_join_mlfq.py"
)
mlfq_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlfq_mod)
_spec = importlib.util.spec_from_file_location(
    "evict_policy", HERE.parent / "python/sglang/srt/mem_cache/evict_policy.py"
)
evict_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evict_mod)


class FakeNode:
    def __init__(self, parent, last_access_time):
        self.parent = parent
        self.last_access_time = last_access_time
        self.mlfq_claim = 0
        self.mlfq_claim_step = -1


class FakeReq:
    def __init__(self, prompt_len: int, cached: int = 0):
        self.origin_input_ids = [0] * prompt_len
        self.num_matched_prefix_tokens = cached
        self.priority = None
        self.mlfq_level = None
        self.mlfq_service_s = 0.0
        self.mlfq_level_entry_time = 0.0
        self.mlfq_last_service_time = 0.0
        self.last_node = None


def test_cost_model_matches_profile():
    model = mlfq_mod.PrefillCostModel(str(PROFILE))
    # Measured 107.5 ms and 171.7 ms, minus the 19.8 ms request overhead.
    assert abs(model.predict_s(uncached=1024, cached=0) - 0.0877) < 1e-3
    assert abs(model.predict_s(uncached=1024, cached=16384) - 0.1519) < 1e-3


def test_cache_aware_places_returning_turn_higher():
    turn = dict(prompt_len=5000, cached=4500)  # a returning turn: long history, short new part
    plain = mlfq_mod.SkipJoinMlfq(cost_mode="prompt", profile_path=str(PROFILE))
    aware = mlfq_mod.SkipJoinMlfq(cost_mode="cache-aware", profile_path=str(PROFILE))
    a, b = FakeReq(**turn), FakeReq(**turn)
    plain.order([a])
    aware.order([b])
    print(f"returning turn: prompt-cost level {a.mlfq_level}, cache-aware level {b.mlfq_level}")
    assert a.mlfq_level == 6 and b.mlfq_level == 3
    assert a.priority == -6 and b.priority == -3


def test_demotion_after_quantum():
    q = mlfq_mod.SkipJoinMlfq(cost_mode="cache-aware", profile_path=str(PROFILE))
    req = FakeReq(prompt_len=5000, cached=4500)
    q.order([req])
    assert req.mlfq_level == 3  # quantum 80 ms
    q.charge([req], 0.050)
    assert req.mlfq_level == 3 and abs(req.mlfq_service_s - 0.050) < 1e-9
    q.charge([req], 0.040)
    assert req.mlfq_level == 4 and req.mlfq_service_s == 0.0 and req.priority == -4


def test_order_by_level_then_entry_time():
    q = mlfq_mod.SkipJoinMlfq(cost_mode="prompt", profile_path=str(PROFILE))
    long_first = FakeReq(prompt_len=8000)
    short_old, short_new = FakeReq(prompt_len=256), FakeReq(prompt_len=256)
    queue = [long_first, short_old]
    q.order(queue)
    time.sleep(0.001)
    queue.append(short_new)
    q.order(queue)
    assert queue == [short_old, short_new, long_first]


def test_starvation_promotion():
    q = mlfq_mod.SkipJoinMlfq(cost_mode="prompt", profile_path=str(PROFILE))
    starved = FakeReq(prompt_len=8000)
    queue = [starved]
    q.order(queue)
    assert starved.mlfq_level > 0
    starved.mlfq_last_service_time -= mlfq_mod.STARVATION_S + 1
    for _ in range(mlfq_mod.STARVATION_CHECK_INTERVAL):
        q.order(queue)
    assert starved.mlfq_level == 0 and starved.priority == 0


def test_two_tier_eviction_order():
    strategy = evict_mod.MlfqTierStrategy()
    root = FakeNode(None, 0)
    shared = FakeNode(root, 1)            # prefix shared by two waiting requests
    waiting_hi = FakeNode(shared, 2)      # level-1 request's own tail
    waiting_lo = FakeNode(shared, 3)      # level-4 request's own tail
    finished_old, finished_new = FakeNode(root, 4), FakeNode(root, 5)
    nodes = {1: shared, 2: waiting_hi, 3: waiting_lo}

    q = mlfq_mod.SkipJoinMlfq(cost_mode="cache-aware", profile_path=str(PROFILE),
                              claim_strategy=strategy, node_of=nodes.__getitem__)
    hi, lo = FakeReq(prompt_len=300), FakeReq(prompt_len=8000)
    hi.last_node, lo.last_node = 2, 3
    q.order([hi, lo])
    assert hi.mlfq_level < lo.mlfq_level

    order = sorted([shared, waiting_hi, waiting_lo, finished_old, finished_new],
                   key=strategy.get_priority)
    # Unclaimed (finished) KV first by recency, then claimed KV lowest level first;
    # the shared prefix is protected at the higher-priority request's level.
    assert order == [finished_old, finished_new, waiting_lo, shared, waiting_hi]
    assert shared.mlfq_claim == hi.priority

    strategy.begin_step()  # the next step, with nobody waiting: every claim expires
    assert all(strategy.get_priority(n)[0] == 0 for n in (shared, waiting_hi, waiting_lo))


class FakeHiCacheTree:
    """The slice of UnifiedRadixCache that MlfqLoadAhead uses; loads finish on ack()."""

    def __init__(self, free_tokens: int):
        self.ongoing_load_back = {}
        self.cache_controller = types.SimpleNamespace(ack_load_queue=[])
        self.token_to_kv_pool_allocator = types.SimpleNamespace(
            size=100_000, available_size=lambda: free_tokens)
        self.loaded = []

    def evictable_size(self):
        return 0

    def init_load_back(self, params):
        self.loaded.append(params.best_match_node)
        self.ongoing_load_back[params.best_match_node] = True
        return params.host_hit_length, params.best_match_node

    def ready_to_load_host_cache(self):
        return 2  # the layer-counter slot the copy was started in

    def ack(self):
        self.ongoing_load_back.clear()


def _install_fake_sglang_helpers():
    """MlfqLoadAhead imports SGLang's load-back params and prefix matcher lazily."""
    base = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    base.InitLoadBackParams = lambda **kw: types.SimpleNamespace(**kw)
    common = types.ModuleType("sglang.srt.mem_cache.common")
    common.match_kv_cache = lambda req, tree: None  # host_hit_length is preset below
    sys.modules[base.__name__], sys.modules[common.__name__] = base, common


def test_load_ahead_copies_next_waiting_prefixes():
    _install_fake_sglang_helpers()
    tree = FakeHiCacheTree(free_tokens=60_000)  # budget: min(32768, 60000 - 10% of pool)
    ahead = mlfq_mod.MlfqLoadAhead(tree_cache=tree)

    def waiting(node, host_hit):
        return types.SimpleNamespace(best_match_node=node, host_hit_length=host_hit)

    # In MLFQ order: on the GPU already, two CPU-only prefixes, one too small to
    # be worth it, then one that would push the copy past its 32k-token cap.
    queue = [waiting(1, 0), waiting(2, 12_000), waiting(3, 100), waiting(4, 16_000),
             waiting(5, 8_000), waiting(6, 1_000)]
    ahead.step(queue)
    assert tree.loaded == [2, 4]  # in waiting order, stopping at the first that does not fit
    assert ahead.consumer_index() == 2  # a prefill admitted now must wait on the copy

    ahead.step(queue)  # one copy at a time
    assert tree.loaded == [2, 4]

    tree.ack()
    assert ahead.consumer_index() == -1
    ahead.step([waiting(5, 8_000)])
    assert tree.loaded == [2, 4, 5]


def test_online_cost_model_starts_from_seed_and_learns():
    model = mlfq_mod.OnlinePrefillCostModel()
    # Seed: about 0.6 s for an 8k chunk with no prefix, 1.3 s on a 32k prefix (A100, 8B).
    assert 0.5 < model.predict_s(uncached=8192, cached=0) < 0.7
    assert 1.2 < model.predict_s(uncached=8192, cached=32768) < 1.5

    # A GPU twice as slow: feed it batches with doubled times and check it adapts.
    def truth(rows):
        return 2 * sum(mlfq_mod.OnlinePrefillCostModel().predict_s(uncached=u, cached=c)
                       for u, c in rows)
    batches = [[(8192, 0)], [(2048, 32768), (1024, 0)], [(8192, 98304)], [(512, 4096)] * 4,
               [(4096, 16384)], [(1, 20000)] * 8 + [(2048, 0)]]
    for _ in range(5):
        for rows in batches:
            model.observe(rows, truth(rows))
    for u, c in [(8192, 0), (2048, 32768), (512, 98304)]:
        want = truth([(u, c)])
        assert abs(model.predict_s(uncached=u, cached=c) - want) < 0.15 * want, (u, c)
    model.observe([(8192, 0)], 500.0)  # an implausible time is ignored
    assert model.observations == 30


def test_mlfq_without_profile_places_with_the_online_model():
    q = mlfq_mod.SkipJoinMlfq(cost_mode="cache-aware", profile_path=None)
    turn, fresh = FakeReq(prompt_len=5000, cached=4500), FakeReq(prompt_len=60000)
    q.order([fresh, turn])
    assert turn.mlfq_level < fresh.mlfq_level
    assert turn.mlfq_level == 3 and fresh.mlfq_level == 9  # 0.045 s vs 4.3 s predicted


def test_chunk_limit_reserves_for_more_urgent_waiting_requests():
    limit = mlfq_mod.mlfq_chunk_limit
    # Levels 3 and 4 are more urgent than the level-8 chunked request; level 9 is not.
    waiting = [(3, 512), (4, 1024), (9, 100)]
    assert limit(chunked_level=8, waiting=waiting, budget=4096, page_size=256, gate_level=None) == 2560
    # The gate admits only level <= 3 this step, so only the first is reserved for.
    assert limit(chunked_level=8, waiting=waiting, budget=4096, page_size=256, gate_level=3) == 3584
    # Reservation rounds up to pages and always leaves the chunk one page of progress.
    assert limit(chunked_level=8, waiting=[(1, 257)], budget=4096, page_size=256, gate_level=None) == 3584
    assert limit(chunked_level=8, waiting=[(1, 3840)], budget=4096, page_size=256, gate_level=None) == 256
    for kwargs in [dict(chunked_level=3, waiting=waiting), dict(chunked_level=None, waiting=waiting),
                   dict(chunked_level=8, waiting=[(None, 10)]), dict(chunked_level=8, waiting=[(2, 4096)]),
                   dict(chunked_level=8, waiting=[]), dict(chunked_level=8, waiting=waiting, budget=256)]:
        kwargs = dict(budget=4096, page_size=256, gate_level=None) | kwargs
        assert limit(**kwargs) is None, kwargs


def _load_prefill_budget():
    swa = types.ModuleType("sglang.srt.mem_cache.allocator.swa")
    swa.is_swa_req_ring = lambda *a, **k: False
    sys.modules[swa.__name__] = swa
    spec = importlib.util.spec_from_file_location(
        "prefill_budget", HERE.parent / "python/sglang/srt/mem_cache/prefill_budget.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_continuing_chunk_never_exceeds_physical_tokens():
    # Regression: with the reservation budget exhausted, a chunked prefill used to be
    # handed its full chunk and crash in alloc_token_slots ("Out of memory. Try to
    # lower your batch size") when the pool held fewer tokens than the chunk.
    budget_mod = _load_prefill_budget()
    allocator = types.SimpleNamespace(page_size=1, available_size=lambda: 7670)
    tree = types.SimpleNamespace(supports_mamba=lambda: False, evictable_size=lambda: 0)
    budget = budget_mod.PrefillBudget(allocator, tree, num_mixed_decode_tokens=31)
    budget.total_offset += 20_000  # running requests' reserved decode headroom
    assert budget.remaining_total < 0
    assert budget.available_chunk_tokens(8161) == 7669  # what physically fits, one page spare
    allocator.available_size = lambda: 0
    assert budget.available_chunk_tokens(8161) == 1  # still asks for progress, not a crash-sized chunk
    budget.total_offset = 31
    allocator.available_size = lambda: 50_000
    assert budget.available_chunk_tokens(8161) == 8161  # budget has room: unchanged behaviour


def test_prefill_gate():
    def at(level):
        return types.SimpleNamespace(mlfq_level=level)

    gate = mlfq_mod.SkipJoinMlfq.prefill_gate_level
    assert gate([at(2), at(7)]) == 2  # a level-2 request is mid-answer
    assert gate([at(None), at(4)]) == 4
    assert gate([]) is None  # nothing running: prefill whatever waits

    # The admission loop stops at the first waiting request past the gate, so a
    # short head no longer lets a long, low-priority prefill in behind it.
    waiting, limit = [at(1), at(2), at(9), at(3)], gate([at(2), at(7)])
    admitted = []
    for req in waiting:
        if req.mlfq_level > limit:
            break
        admitted.append(req.mlfq_level)
    assert admitted == [1, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
