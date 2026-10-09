"""Isolate eager-vs-graph DECODE logit differences.

Prefills once (eager). Then, for each decode step, runs the SAME step in eager
and graph mode from an identical KV-cache snapshot, comparing logits. The KV
snapshot fed to both modes is always written by the eager path, so the graph
path is tested in isolation (its own store_kvcache + flash_attn_with_kvcache).

Usage:
  PYTHONPATH=/home/Tokn .venv/bin/python debug/graphdiag.py --prompt-line 5 --steps 48
"""

import argparse
import copy
import sys

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import reset_context


def load_prompt(line_no: int) -> str:
    with open("data/prefix_cache_test_prompts.txt", "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines[line_no]


def tokenize(run: Run, prompt_text: str) -> list[int]:
    text = run.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return run.tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()


def build_seq(run: Run, token_ids: list[int], max_tokens: int = 48):
    seq = Sequence(token_ids, max_tokens=max_tokens)
    alloc_token_ids = token_ids + [0] * max_tokens
    cached_blocks = run.block_manager.match_prefix(token_ids)
    block_table_ids = run.block_manager.allocate(
        token_ids=alloc_token_ids, num_cached_blocks=cached_blocks
    )
    seq.block_table_ids = block_table_ids
    seq.block_table = run.block_manager.make_block_table_tensor(
        block_table_ids, torch.device(run.device)
    )
    seq.num_cached_tokens = cached_blocks * run.block_manager.block_size
    seq.num_hashed_blocks = cached_blocks
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    return seq


def run_prefill(run: Run, seq: Sequence):
    with torch.inference_mode():
        input_ids, positions = run.prepare_prefill([seq])
        logits = run.run_model(input_ids, positions, is_prefill=True)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        tok = int(torch.argmax(logits[0]).item())
        seq.append_token(tok)
        run.block_manager.hash_completed_blocks(
            token_ids=seq.token_ids,
            block_table=seq.block_table_ids,
            start_block=seq.num_hashed_blocks,
            end_block=len(seq.token_ids) // run.block_size,
        )
        seq.num_hashed_blocks = len(seq.token_ids) // run.block_size
    return logits.detach().float().cpu()


def decode_step(run: Run, seq: Sequence, use_graph: bool):
    run.enforce_eager = not use_graph
    with torch.inference_mode():
        input_ids, positions = run.prepare_decode([seq])
        logits = run.run_model(input_ids, positions, is_prefill=False)
        tok = int(torch.argmax(logits[0]).item())
        seq.append_token(tok)
    return logits.detach().float().cpu(), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-line", type=int, default=5)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_length=2048,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=False,  # graphs captured
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])

    prompt_text = load_prompt(args.prompt_line)
    token_ids = tokenize(run, prompt_text)
    print(f"prompt tokens: {len(token_ids)}", flush=True)

    # --- prefill once (eager) ---
    seq = build_seq(run, token_ids, max_tokens=args.steps)
    prefill_logit = run_prefill(run, seq)
    print(f"prefill logit argmax = {int(torch.argmax(prefill_logit))}", flush=True)

    # --- decode steps, eager vs graph from identical KV snapshot ---
    base_kv = run.kv_cache.clone()
    seq_g = copy.deepcopy(seq)  # same length, same block table
    seq_g.token_ids = list(seq.token_ids)

    logs_e = [prefill_logit]
    logs_g = [prefill_logit.clone()]
    toks_e = [int(torch.argmax(prefill_logit))]
    toks_g = [int(torch.argmax(prefill_logit))]

    for i in range(args.steps):
        # eager step writes canonical next K/V
        run.kv_cache.copy_(base_kv)
        le, te = decode_step(run, seq, use_graph=False)
        logs_e.append(le)
        toks_e.append(te)
        base_kv = run.kv_cache.clone()

        # graph step reads the same canonical KV, tests graph decode alone
        run.kv_cache.copy_(base_kv)
        lg, tg = decode_step(run, seq_g, use_graph=True)
        logs_g.append(lg)
        toks_g.append(tg)

        md = (le - lg).abs().max().item()
        rel = md / (le.abs().max().item() + 1e-9)
        flag = "  <-- DIVERGED" if (md > 0.1 or te != tg) else ""
        print(f"step {i+1:3d} max|d|= {md:.4g} rel= {rel:.2e} tok_e={te} tok_g={tg}{flag}",
              flush=True)
        torch.cuda.synchronize()

    print(f"token diffs (excl prefill): "
          f"{sum(1 for a, b in zip(toks_e[1:], toks_g[1:]) if a != b)}/{args.steps}", flush=True)

    run.block_manager.deallocate(seq.block_table_ids)

    if args.dump:
        torch.save({"logs_e": logs_e, "logs_g": logs_g}, args.dump)


if __name__ == "__main__":
    main()
