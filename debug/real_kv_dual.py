"""Test: at ctx 257-369 with REAL model KV, are the fast (ns=1) and splitkv (ns=0)
decode graphs bit-identical? Run both graphs on the same live cache state and compare."""
import sys
sys.path.insert(0, "/home/Tokn")
import torch as T
from src.llm import LLM
from utils.sampling_params import SamplingParams
from utils.configs import ServerConfigs
from utils.context import reset_context

cfg = ServerConfigs(
    model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda",
    max_length=2048, max_model_len=2048, tensor_parallel_size=1,
    enforce_eager=False, max_num_seqs=5, max_num_batched_tokens=1024,
)
llm = LLM(model="Qwen/Qwen3-0.6B", dtype="bf16", device="cuda", max_model_len=2048, config=cfg)
RUN = llm.RUN

# Build a sequence with a long prompt (>= 300 tokens) via the tokenizer
tok = llm.tokenizer
text = "The quick brown fox jumps over the lazy dog. " * 14  # ~120 tokens
ids = tok(text, add_special_tokens=False).input_ids
# pad to ~300 tokens
while len(ids) < 300:
    ids = ids + ids
ids = ids[:300]
print("prompt len:", len(ids))

from utils.sequence import Sequence
seq = Sequence(token_ids=ids, max_tokens=400)
# allocate blocks
alloc = ids + [0]*400
bt = llm.block_manager.allocate(token_ids=alloc, num_cached_blocks=0)
seq.block_table_ids = bt
seq.block_table = llm.block_manager.make_block_table_tensor(block_table=bt, device="cuda")
seq.num_cached_tokens = 0

# prefill via engine.step path
seq.num_scheduled_tokens = len(ids)
with T.inference_mode():
    RUN.call("run", [seq], True)  # prefill -> writes KV for all 300 tokens
    seq.num_cached_tokens = 300
    seq.num_scheduled_tokens = 0
    print("after prefill cached:", seq.num_cached_tokens, "last:", seq.last_token)

    # Now decode, at each ctx in [257,369] run BOTH graphs and compare tokens
    ndiff = 0
    ntest = 0
    mismatches = []
    for step in range(120):
        ctx = len(seq.token_ids)
        if ctx > 369:
            break
        if ctx >= 257:
            # run fast graph, then splitkv graph on the SAME state
            input_ids, positions = RUN.prepare_decode([seq])
            # need to reset the token each time (in-graph argmax writes tbuf_tok)
            RUN.graph_vars["tbuf_tok"][:1].copy_(T.tensor([seq.last_token], dtype=T.int32, device="cuda"))
            reset_context()
            # fast
            RUN.prepare_decode([seq], token_only=True)
            try:
                RUN.run_model(input_ids, positions, False)
            finally:
                reset_context()
            tok_fast = int(RUN.graph_vars["tbuf_tok"][0].item())
            # reset the token buffer (the fast graph's argmax wrote it)
            RUN.graph_vars["tbuf_tok"][:1].copy_(T.tensor([seq.last_token], dtype=T.int32, device="cuda"))
            # splitkv
            RUN.prepare_decode([seq], token_only=True)
            try:
                RUN.run_model(input_ids, positions, False)
            finally:
                reset_context()
            tok_split = int(RUN.graph_vars["tbuf_tok"][0].item())
            ntest += 1
            if tok_fast != tok_split:
                ndiff += 1
                mismatches.append((ctx, tok_fast, tok_split))
        # advance: append the splitkv token (the reference)
        nid = int(RUN.graph_vars["tbuf_tok"][0].item())
        seq.append_token(nid)
    print(f"tested {ntest} steps in ctx 257-369: {ndiff} token mismatches fast-vs-splitkv")
    print("mismatches:", mismatches[:10])
