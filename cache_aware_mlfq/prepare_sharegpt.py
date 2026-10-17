"""Build the realistic workload: real multi-turn ShareGPT conversations that fit the GPU pool.

SGLang's ShareGPT replay (benchmark/hicache/bench_serving.py --enable-multiturn)
does not cap a conversation's total length (--sharegpt-context-len is parsed
but unused), and a conversation longer than the 24,576-token GPU KV pool cannot
run. This keeps the conversations whose whole history fits, in ShareGPT's own
format, so the benchmark loads it unchanged with --dataset-path. Runs on the pod.

    python cache_aware_mlfq/prepare_sharegpt.py
"""

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from sglang.benchmark.datasets.common import SHAREGPT_FILENAME, SHAREGPT_REPO_ID
from sglang.benchmark.utils import download_and_cache_hf_file

MODEL = "Qwen/Qwen3-8B"
# Leaves room in the 24,576-token GPU pool for the chat template and other requests.
MAX_CONVERSATION_TOKENS = 16000
# The benchmark drops shorter turns, which would break a conversation's structure.
MIN_MESSAGE_TOKENS = 4
OUT = Path("/workspace/data/sharegpt_multiturn_16k.json")


def well_formed(messages: list) -> bool:
    if len(messages) < 2 or len(messages) % 2:
        return False
    return all(m["from"] == ("human" if i % 2 == 0 else "gpt") for i, m in enumerate(messages))


def percentile(values: list, q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    path = download_and_cache_hf_file(repo_id=SHAREGPT_REPO_ID, filename=SHAREGPT_FILENAME)
    with open(path) as f:
        dataset = json.load(f)
    candidates = [c for c in dataset if well_formed(c.get("conversations", []))]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    texts = [m["value"] for c in candidates for m in c["conversations"]]
    lengths = []
    for start in range(0, len(texts), 4096):
        batch = tokenizer(texts[start:start + 4096], add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in batch)

    kept, turns, replies, totals = [], [], [], []
    pos = 0
    for conv in candidates:
        n = len(conv["conversations"])
        lens, pos = lengths[pos:pos + n], pos + n
        if min(lens) < MIN_MESSAGE_TOKENS or sum(lens) > MAX_CONVERSATION_TOKENS:
            continue
        kept.append(conv)
        turns.append(n // 2)
        replies.extend(lens[1::2])
        totals.append(sum(lens))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept))
    multi = sum(t >= 2 for t in turns)
    print(f"source conversations: {len(dataset)}, well-formed: {len(candidates)}, kept: {len(kept)}")
    print(f"turns per conversation: mean {statistics.mean(turns):.2f}, "
          f"multi-turn share {multi / len(turns):.0%}, p90 {percentile(turns, 0.9)}, max {max(turns)}")
    print(f"reply tokens: mean {statistics.mean(replies):.0f}, median {percentile(replies, 0.5)}, "
          f"p90 {percentile(replies, 0.9)}, p99 {percentile(replies, 0.99)}, max {max(replies)}")
    print(f"conversation tokens: mean {statistics.mean(totals):.0f}, p90 {percentile(totals, 0.9)}, "
          f"max {max(totals)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
