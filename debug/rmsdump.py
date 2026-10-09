"""Dump prefill+decode hidden states for a prompt to compare old vs new RMSNorm."""
import sys, json
sys.path.insert(0, "/home/Tokn")
import torch
from utils.configs import ServerConfigs
from src.run import Run
from utils.sequence import Sequence
from utils.context import reset_context

def load_prompt(line_no):
    with open("data/prefix_cache_test_prompts.txt") as f:
        lines=[ln.rstrip("\n") for ln in f if ln.strip()]
    return lines[line_no]

def tokenize(run, text):
    t = run.tokenizer.apply_chat_template([{"role":"user","content":text}], tokenize=False, add_generation_prompt=True)
    return run.tokenizer([t], return_tensors="pt")["input_ids"][0].tolist()

def build_seq(run, token_ids, max_tokens=48):
    seq = Sequence(token_ids, max_tokens=max_tokens)
    alloc = token_ids + [0]*max_tokens
    cb = run.block_manager.match_prefix(token_ids)
    bt = run.block_manager.allocate(token_ids=alloc, num_cached_blocks=cb)
    seq.block_table_ids = bt
    seq.block_table = run.block_manager.make_block_table_tensor(bt, torch.device(run.device))
    seq.num_cached_tokens = cb * run.block_manager.block_size
    seq.num_hashed_blocks = cb
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    return seq

def main():
    cfg = ServerConfigs(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
        max_length=2048, max_model_len=2048, tensor_parallel_size=1,
        enforce_eager=True, max_num_seqs=5, max_num_batched_tokens=1024)
    run = Run(cfg, 0, [])
    seq = build_seq(run, tokenize(run, load_prompt(0)), max_tokens=48)
    seq.num_scheduled_tokens = seq.num_tokens
    # prefill
    with torch.inference_mode():
        inp, pos = run.prepare_prefill([seq])
        logits = run.run_model(inp, pos, True)
        tok = int(torch.argmax(logits[0]))
        seq.append_token(tok)
        run.block_manager.hash_completed_blocks(token_ids=seq.token_ids,
            block_table=seq.block_table_ids, start_block=seq.num_hashed_blocks,
            end_block=len(seq.token_ids)//run.block_size)
        seq.num_hashed_blocks = len(seq.token_ids)//run.block_size
    # capture prefill first-layer inputs? Instead: dump per-layer outputs of prefill model
    hs = run.custom_model.model.embed_tokens(inp)
    layer_out = []
    residual = None
    with torch.inference_mode():
        for layer in run.custom_model.model.layers:
            hs, residual = layer(pos, hs, residual)
            layer_out.append(hs.detach().float().cpu())
        final, _ = run.custom_model.model.norm(hs, residual)
    torch.save({"prefill_layers": layer_out, "final": final.detach().float().cpu()}, "/tmp/rms_prefill.pt")
    print("dumped prefill layer states", flush=True)

if __name__ == "__main__":
    main()
