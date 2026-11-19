"""
Sample a fixed set of real user prompts from ShareGPT for benchmarking.

Writes JSONL in the same {"question": ...} format as data/questions.jsonl, so
benchmark.py --questions <out> works unchanged and tokn and vLLM see exactly
the same prompts in the same order.

ShareGPT is the real-chat dataset used by vLLM's own serving benchmarks.
We take the first human turn of each conversation, keep prompts whose
chat-templated length fits in max_model_len - max_tokens, shuffle with a
fixed seed and take the first N.

Usage:
    python debug/make_sharegpt.py --num-prompts 400 --out data/sharegpt_400.jsonl
"""

import argparse
import json
import random
import re

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

# A prompt pasted a live SendGrid key; GitHub push protection rejects it.
SENDGRID_KEY = re.compile(r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num-prompts", type=int, default=400)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--min-prompt-tokens", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/sharegpt_400.jsonl")
    args = p.parse_args()

    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_prompt_tokens = args.max_model_len - args.max_tokens

    random.Random(args.seed).shuffle(data)

    picked = []
    for conv in data:
        turns = conv.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        q = (turns[0].get("value") or "").strip()
        if not q:
            continue
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n < args.min_prompt_tokens or n > max_prompt_tokens:
            continue
        picked.append({"question": q, "prompt_tokens": n, "id": conv.get("id")})
        if len(picked) == args.num_prompts:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        for row in picked:
            row["question"] = SENDGRID_KEY.sub("SG.REDACTED", row["question"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = sorted(r["prompt_tokens"] for r in picked)
    print(f"wrote {len(picked)} prompts to {args.out}")
    print(f"prompt tokens: min={lens[0]} p50={lens[len(lens) // 2]} max={lens[-1]} "
          f"total={sum(lens)}")


if __name__ == "__main__":
    main()
