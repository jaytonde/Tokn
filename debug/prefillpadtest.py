"""Test: is flash_attn_varlen prefill bit-identical when total tokens are padded
to a bucket (padding = separate varlen sequence masked via cu_seqlens)?

Compares full-model prefill logits: unpadded real prefill vs padded-to-bucket.
"""
import sys
import time

sys.path.insert(0, "/home/Tokn")

import torch

from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import get_context, set_context, reset_context


def main() -> None:
    cfg = ServerConfigs(
        model="Qwen/Qwen3-0.6B",
        dtype="bf16",
        device="cuda",
        max_length=2048,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_num_seqs=5,
        max_num_batched_tokens=1024,
    )
    run = Run(cfg, 0, [])
    dev = torch.device("cuda")

    import json
    texts = []
    with open("data/questions.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["question"])
            if len(texts) >= 5:
                break

    def tokenize(text):
        t = run.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                              tokenize=False, add_generation_prompt=True)
        return run.tokenizer([t], return_tensors="pt")["input_ids"][0].tolist()

    def build_seq(token_ids, max_tokens=256):
        seq = Sequence(token_ids, max_tokens=max_tokens)
        alloc = token_ids + [0] * max_tokens
        cb = run.block_manager.match_prefix(token_ids)
        bt = run.block_manager.allocate(token_ids=alloc, num_cached_blocks=cb)
        seq.block_table_ids = bt
        seq.block_table = run.block_manager.make_block_table_tensor(bt, torch.device(run.device))
        seq.num_cached_tokens = cb * run.block_manager.block_size
        seq.num_hashed_blocks = cb
        seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
        return seq

    seqs = [build_seq(tokenize(t)) for t in texts]
    lengths = [len(s.token_ids) for s in seqs]
    print(f"lengths={lengths}")

    with torch.inference_mode():
        reset_context()
        input_ids, positions = run.prepare_prefill(seqs[:5])
        S = input_ids.numel()
        ctx = get_context()
        logits_e = run.run_model(input_ids, positions, True)
        torch.cuda.synchronize()
        print(f"unpadded S={S} eager logits shape={tuple(logits_e.shape)}")

        # Pad to buckets: append a fake varlen sequence of padding tokens
        for bucket in (64, 128, 192, 256, 512):
            pad = bucket - S
            if pad <= 0:
                continue
            pad_ids = torch.zeros(pad, dtype=torch.long, device=dev)
            pad_pos = torch.arange(S, S + pad, dtype=torch.long, device=dev)
            in_p = torch.cat([input_ids, pad_ids])
            pos_p = torch.cat([positions, pad_pos])
            # cu_seqlens: real + pad seq (pad starts at S, length bucket-S)
            cu_q = ctx.cu_seqlens_q.clone()
            cu_p = torch.cat([cu_q, torch.tensor([bucket], dtype=torch.int32, device=dev)])
            cu_k = cu_p.clone()
            # block_table: real rows + a padding row (use row 0's table, padded)
            bt_p = torch.cat([ctx.block_tables, ctx.block_tables[:1].clone()], dim=0)
            # slot_mapping: real + pad slots (-1)
            slot_p = torch.cat([ctx.slot_mapping, torch.full((pad,), -1, dtype=torch.int32, device=dev)])
            set_context(is_prefill=True, cu_seqlens_q=cu_p, cu_seqlens_k=cu_k,
                        max_seqlen_q=max(ctx.max_seqlen_q, pad), max_seqlen_k=bucket,
                        slot_mapping=slot_p, block_tables=bt_p)
            logits_p = run.run_model(in_p, pos_p, True)
            torch.cuda.synchronize()
            eq = torch.equal(logits_e, logits_p[:len(seqs[:5])])
            md = (logits_e.float() - logits_p[:len(seqs[:5])].float()).abs().max().item()
            print(f"pad S={S}->{bucket}: real-token logits bit_eq={eq} max|d|={md:.3e}")

        reset_context()


if __name__ == "__main__":
    main()
