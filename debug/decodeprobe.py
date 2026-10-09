"""Isolate eager vs graph decode: dump decode logits after an identical prefill.

Usage: python decodeprobe.py --mode eager|graph --out /tmp/logits_eager.pt
"""

import argparse
import sys

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import reset_context


def build_seq(run, token_ids, max_tokens):
    seq = Sequence(token_ids, max_tokens=max_tokens)
    alloc_token_ids = token_ids + [0] * max_tokens
    cached_blocks = run.block_manager.match_prefix(token_ids)
    block_table_ids = run.block_manager.allocate(
        token_ids=alloc_token_ids, num_cached_blocks=cached_blocks
    )
    seq.block_table_ids = block_table_ids
    seq.block_table = run.block_manager.make_block_table_tensor(block_table_ids, torch.device(run.device))
    seq.num_cached_tokens = cached_blocks * run.block_manager.block_size
    seq.num_hashed_blocks = cached_blocks
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "graph"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_length=2048,
        max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=(args.mode == "eager"),
        max_num_seqs=5, max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])

    torch.manual_seed(1234)
    prompt = torch.randint(0, 151000, (80,)).tolist()

    seq = build_seq(run, prompt, max_tokens=32)
    seq.num_scheduled_tokens = seq.num_tokens

    # Prefill (eager in both modes)
    input_ids, positions = run.prepare_prefill([seq])
    with torch.inference_mode():
        logits = run.run_model(input_ids, positions, is_prefill=True)
    tok = torch.argmax(logits[0]).item()
    seq.append_token(tok)
    run.block_manager.hash_completed_blocks(
        token_ids=seq.token_ids,
        block_table=seq.block_table_ids,
        start_block=seq.num_hashed_blocks,
        end_block=len(seq.token_ids) // run.block_size,
    )
    seq.num_hashed_blocks = len(seq.token_ids) // run.block_size

    decode_logits = []
    decode_tokens = []
    for _ in range(args.steps):
        input_ids, positions = run.prepare_decode([seq])
        with torch.inference_mode():
            logits = run.run_model(input_ids, positions, is_prefill=False)
        decode_logits.append(logits.detach().float().cpu())
        tok = torch.argmax(logits[0]).item()
        decode_tokens.append(tok)
        seq.append_token(tok)
        # sync before next step so timing isn't the point here
        torch.cuda.synchronize()

    torch.save({"logits": decode_logits, "tokens": decode_tokens, "mode": args.mode}, args.out)
    print(f"{args.mode}: decoded tokens = {decode_tokens}")


if __name__ == "__main__":
    main()
