"""ctx 257-369 REAL KV: fast(ns=1) vs splitkv(ns=0) decode graphs bit-identical?"""
import sys
sys.path.insert(0, "/home/Tokn")
import torch as T
from src.llm import LLM
from utils.configs import ServerConfigs
from utils.context import reset_context, set_context

cfg = ServerConfigs(
    model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
    max_length=2048, max_model_len=2048, tensor_parallel_size=1,
    enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
)
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_model_len=2048, config=cfg)
RUN = llm.RUN
tok = llm.tokenizer

text = "The quick brown fox jumps over the lazy dog. " * 14
ids = tok(text, add_special_tokens=False).input_ids
while len(ids) < 300:
    ids = ids + ids
ids = ids[:300]

from utils.sequence import Sequence
seq = Sequence(token_ids=ids, max_tokens=400)
bt = llm.block_manager.allocate(token_ids=ids + [0]*400, num_cached_blocks=0)
seq.block_table_ids = bt
seq.block_table = llm.block_manager.make_block_table_tensor(block_table=bt, device="cuda")
seq.num_scheduled_tokens = len(ids)
with T.inference_mode():
    RUN.call("run", [seq], True)
    seq.num_cached_tokens = 300
    seq.num_scheduled_tokens = 0

    gv = RUN.graph_vars
    ndiff = ntest = 0
    mismatches = []
    for step in range(140):
        ctx = len(seq.token_ids)
        if ctx > 369:
            break
        if ctx >= 257:
            # full prep sets context for the current step (fast flag from ctx<=256 -> False here)
            input_ids, positions = RUN.prepare_decode([seq])
            reset_context()
            # run fast: override context flag
            for run_fast, label in ((True, "fast"), (False, "split")):
                set_context(is_prefill=False, slot_mapping=gv["g_slot"][:1],
                            context_lens=gv["g_ctx"][:1], block_tables=gv["block_tables"][:1],
                            use_fast_flash=run_fast)
                gv["tbuf_tok"][:1].copy_(T.tensor([seq.last_token], dtype=T.int32, device="cuda"))
                T.cuda.synchronize()
                try:
                    RUN.run_model(input_ids, positions, False)
                finally:
                    reset_context()
                t = int(gv["tbuf_tok"][0].item())
                if run_fast:
                    tok_fast = t
                else:
                    tok_split = t
            ntest += 1
            if tok_fast != tok_split:
                ndiff += 1
                mismatches.append((ctx, tok_fast, tok_split))
        # advance with the splitkv (reference) token
        nid = int(gv["tbuf_tok"][0].item())
        seq.append_token(nid)
    print(f"ctx 257-369 tested {ntest} steps: {ndiff} fast-vs-splitkv token mismatches")
    print("mismatches:", mismatches[:12])
