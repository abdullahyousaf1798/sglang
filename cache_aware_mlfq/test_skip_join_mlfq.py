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
