"""Unit checks for managers/skip_join_mlfq.py; no GPU or SGLang install needed.

    python3 cache_aware_mlfq/test_skip_join_mlfq.py
"""

import importlib.util
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "prefill_profile_4090.json"
_spec = importlib.util.spec_from_file_location(
    "skip_join_mlfq", HERE.parent / "python/sglang/srt/managers/skip_join_mlfq.py"
)
mlfq_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlfq_mod)


class FakeReq:
    def __init__(self, prompt_len: int, cached: int = 0):
        self.origin_input_ids = [0] * prompt_len
        self.num_matched_prefix_tokens = cached
        self.priority = None
        self.mlfq_level = None
        self.mlfq_service_s = 0.0
        self.mlfq_level_entry_time = 0.0
        self.mlfq_last_service_time = 0.0


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
