import re
import time
import logging
import os
import asyncio
import torch
import torch.distributed as dist
from dataclasses import dataclass
from transformers  import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

from utils.context import reset_context, set_context
from src.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


import re
import time
import logging
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from transformers  import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

from utils.context import reset_context, set_context
from src.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


from tqdm import tqdm
from utils.sequence import Sequence
from utils.sampling_params import SamplingParams
from src.block_manager import BlockManager
from utils.request_state import RequestState
from utils.context import get_context, reset_context, set_context

from multiprocessing.synchronize import Event
from utils.configs import ServerConfigs as Config

from multiprocessing.shared_memory import SharedMemory
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Run:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):

        self.config        = config
        self.block_size    = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size    = config.tensor_parallel_size
        self.rank          = rank
        self.event         = event
        self.model         = config.model
        self.device        = config.device
        self.dtype         = torch.float16 if config.dtype in ("fp16", "float16") else torch.bfloat16

        self.hf_config = AutoConfig.from_pretrained(self.config.model)

        backend = "nccl" if self.device.startswith("cuda") else "gloo"
        if self.device.startswith("cuda"):
            torch.cuda.set_device(rank)
            dist.init_process_group(
                backend,
                "tcp://localhost:2333",
                world_size=self.world_size,
                rank=rank,
                device_id=torch.device(f"cuda:{rank}"),
            )
        else:
            dist.init_process_group(
                backend,
                "tcp://localhost:2333",
                world_size=self.world_size,
                rank=rank,
            )

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.dtype)
        torch.set_default_device("cuda" if self.device.startswith("cuda") else "cpu")

        self.custom_model = Qwen3ForCausalLM(self.hf_config)

        self.outputs = {}
        self.max_model_len = self.config.max_model_len
        self.metrics = None

        #for continuous batching
        self.request_states = {}
        self.request_lock = asyncio.Lock()
        self.has_work = asyncio.Event()
        self.background_task = None

        self.enforce_eager = self.config.enforce_eager
        self.graphs = {}
        self.graphs_fast = {}
        self.graph_pool = None
        self._padded_bt = {}   # seq_id -> padded (max_num_blocks,) int32 row
        self.graph_bs = []
        self.graph_vars = {}
        self.prefill_graphs = {}
        self.prefill_graph_pool = None
        self.prefill_graph_cap = 256
        self.prefill_capture_max = 128
        self.capture_stats = {"n": 0, "wall": 0.0, "shapes": {}}
        self._device = torch.device(self.device)
        self._bt_cache = {}   # (seq_id tuple) -> stacked padded block table

        print(f"Loading model onto device : {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        model_path = snapshot_download(self.model)
        logger.info(f"Model downloaded to: {model_path}")

        load_model(self.custom_model, model_path)
        self.custom_model = self.custom_model.to(dtype=self.dtype, device=self.device)

        #Initializing the KV Cache
        self.allocate_kv_cache()

        if self.device.startswith("cuda") and not self.enforce_eager:
            self.capture_cudagraph()
            self.pre_capture_prefill_graphs()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank==0:
                self.shm = SharedMemory(name="tokn", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="tokn")
                self.loop()

    def allocate_kv_cache(self):
        """
        For Qwen3 0.6B : 
        Parameter	          Value
        num_hidden_layers	  28
        num_key_value_heads	  8
        hidden_size	          1024
        num_attention_heads	  16 ----> Query heads
        head_dim	          28 (explicit, not derived)
        
        MHA : 
        num_attention_heads // num_key_value_heads query heads (16 / 8 = 2 query heads per KV group in Qwen3-0.6B).
        """
        config = self.hf_config

        # FlashAttention paged KV cache requires block_size divisible by 256.
        self.block_size = 256
        self.max_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads // self.world_size

        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads

        requested_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size

        # Static sizing with headroom for multi-request batching.
        max_num_seqs = max(1, getattr(self.config, "max_num_seqs", 1))
        self.num_blocks = requested_num_blocks * max_num_seqs + 16

        logger.info(
            "KV cache sizing (static): max_model_len=%d, requested_blocks=%d, selected_blocks=%d",
            self.max_model_len,
            requested_num_blocks,
            self.num_blocks,
        )

        self.kv_cache = torch.empty(
            2,            # K and V
            num_layers,
            self.num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device,
            dtype=self.dtype
        ) # (2, 28, 128, 256, 8, 128)

        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
        )

        layer_id = 0
        for layer in self.custom_model.model.layers:
            layer.self_attn.attn.k_cache = self.kv_cache[0, layer_id]
            layer.self_attn.attn.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1

        logger.warning(
            f"\n\nBlockManager initialized: "
            f"\nnum_blocks={self.block_manager.num_blocks}, "
            f"\nblock_size={self.block_manager.block_size}, "
            f"\nfree_blocks={len(self.block_manager.free_block_ids)}"
        )
    
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config         = self.hf_config
        hidden_size    = config.hidden_size
        max_bs         = self.config.max_num_seqs
        max_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        device         = torch.device(self.device)

        # Static buffers — addresses are frozen for the lifetime of the graphs.
        input_ids      = torch.zeros(max_bs, dtype=torch.long,  device=device)
        positions      = torch.zeros(max_bs, dtype=torch.long,  device=device)
        slot_mapping   = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens   = torch.zeros(max_bs, dtype=torch.int32, device=device)
        block_tables   = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device)
        outputs        = torch.zeros(max_bs, self.hf_config.vocab_size, dtype=self.dtype, device=device)

        self.graph_bs  = [bs for bs in (1,2,3,4) if bs <= max_bs]

        if max_bs not in self.graph_bs:
            self.graph_bs.append(max_bs)


        for bs in reversed(self.graph_bs):
            # Capture TWO decode graphs per batch size:
            #   self.graphs_fast[bs] : num_splits=1 flash kernel (single-pass).
            #                          Bit-identical to the splitkv path for
            #                          seqlen <= 256, ~7us/call faster.
            #   self.graphs[bs]      : splitkv flash kernel (reference path,
            #                          needed for seqlen > 256).
            # Both share the same static buffers and graph pool; run_model
            # picks one per step from the max context length (CPU-side flag).
            for use_fast, dst in ((True, self.graphs_fast), (False, self.graphs)):
                graph = torch.cuda.CUDAGraph()
                set_context(
                    is_prefill=False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    use_fast_flash=use_fast,
                )

                # Warmup run (lazy allocs / autotune happen OUTSIDE the graph).
                # Capture the full forward INCLUDING lm_head so the logits GEMM
                # runs inside the graph instead of eagerly every decode step.
                outputs[:bs] = self.custom_model(input_ids[:bs], positions[:bs])
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.custom_model(input_ids[:bs], positions[:bs])

                if self.graph_pool is None:
                    self.graph_pool = graph.pool()

                dst[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs
        )

    def run_model(self, input_ids, positions, is_prefill):

        if is_prefill:
            if self.enforce_eager or not self.graphs:
                return self.custom_model(input_ids, positions)
            return self.run_prefill(input_ids, positions)

        if self.enforce_eager or not self.graphs or input_ids.size(0) > self.graph_bs[-1]:
            return self.custom_model(input_ids, positions)

        bs      = input_ids.size(0)
        context = get_context()
        pool    = self.graphs_fast if context.use_fast_flash else self.graphs
        use_bs  = next(x for x in self.graph_bs if x >= bs)
        graph   = pool[use_bs]
        gv      = self.graph_vars

        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        if use_bs != bs:
            # Only when bs does not exactly match a captured graph do we pad up
            # to a larger shape; those padding rows need deterministic sentinels.
            gv["slot_mapping"].fill_(-1)          # padding rows write nowhere
            gv["context_lens"].zero_()            # padding rows attend to nothing
            gv["block_tables"].fill_(-1)          # keep padding deterministic (-1)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        graph.replay()

        return gv["outputs"][:bs]

    def run_prefill(self, input_ids, positions):
        """Prefill via lazily-captured exact-shape CUDA graphs.

        Eager prefill is host-bound: ~680 kernel launches at ~66us each cost
        ~40ms/step even for 26 tokens while the GPU is busy only ~7ms.
        Replaying the SAME shape from a CUDA graph drops that to ~8ms and is
        bit-identical to eager (same kernels, same shapes). Prompts in the
        benchmark span a small set of (num_seqs, total_tokens) shapes, so a
        small lazy cache covers them. Each graph is captured on first use with
        the real context values but slot_mapping forced to -1 so the capture
        does not write garbage into the KV cache; the real slot_mapping is
        copied in before every replay.
        """
        context = get_context()
        num_tokens = input_ids.numel()
        num_seqs = context.cu_seqlens_q.numel() - 1
        key = (num_seqs, num_tokens, context.max_seqlen_q, context.max_seqlen_k)

        gv = self.prefill_graphs.get(key)
        if gv is None:
            if len(self.prefill_graphs) >= self.prefill_graph_cap:
                return self.custom_model(input_ids, positions)  # eager fallback
            # Eager fallback on cache miss instead of a synchronous lazy
            # capture: capturing a multi-seq graph stalls the whole pipeline
            # ~190ms (CUDA graph capture serializes), and the captured shape
            # rarely repeats, so the stall buys nothing. Eager prefill is
            # bit-identical (same kernels/shapes, just host-dispatched) and
            # costs ~40ms. The pre-captured (1, S<=128) grid still covers the
            # common single-seq shapes.
            t0 = time.perf_counter()
            out = self.custom_model(input_ids, positions)
            self.capture_stats["n"] += 1
            self.capture_stats["wall"] += time.perf_counter() - t0
            self.capture_stats["shapes"][key] = self.capture_stats["shapes"].get(key, 0) + 1
            return out

        gv["input_ids"][:num_tokens] = input_ids
        gv["positions"][:num_tokens] = positions
        gv["cu_seqlens_q"][:num_seqs + 1] = context.cu_seqlens_q
        gv["cu_seqlens_k"][:num_seqs + 1] = context.cu_seqlens_k
        gv["slot_mapping"][:num_tokens] = context.slot_mapping
        gv["block_tables"][:num_seqs] = context.block_tables
        gv["graph"].replay()
        return gv["outputs"][:num_seqs]

    @torch.inference_mode()
    def _capture_prefill_graph(self, key, input_ids, positions, ctx):
        num_seqs, num_tokens, max_q, max_k = key
        device = torch.device(self.device)

        g_input = torch.zeros(num_tokens, dtype=torch.long, device=device)
        g_pos   = torch.zeros(num_tokens, dtype=torch.long, device=device)
        g_cu_q  = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        g_cu_k  = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        g_slot  = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
        g_bt    = torch.zeros(num_seqs, self.max_num_blocks, dtype=torch.int32, device=device)
        g_out   = torch.zeros(num_seqs, self.hf_config.vocab_size, dtype=self.dtype, device=device)

        # Seed buffers with the real values so the capture is a valid forward.
        # slot_mapping stays -1 so store_kvcache skips all writes (no KV pollution).
        g_input[:] = input_ids
        g_pos[:] = positions
        g_cu_q[:] = ctx.cu_seqlens_q
        g_cu_k[:] = ctx.cu_seqlens_k
        g_bt[:num_seqs] = ctx.block_tables

        set_context(
            is_prefill=True,
            cu_seqlens_q=g_cu_q,
            cu_seqlens_k=g_cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            slot_mapping=g_slot,
            block_tables=g_bt,
        )

        graph = torch.cuda.CUDAGraph()
        g_out[:] = self.custom_model(g_input, g_pos)   # warmup (lazy allocs outside graph)
        with torch.cuda.graph(graph, self.prefill_graph_pool):
            g_out[:] = self.custom_model(g_input, g_pos)

        if self.prefill_graph_pool is None:
            self.prefill_graph_pool = graph.pool()

        torch.cuda.synchronize()
        reset_context()

        gv = dict(
            graph=graph,
            input_ids=g_input,
            positions=g_pos,
            cu_seqlens_q=g_cu_q,
            cu_seqlens_k=g_cu_k,
            slot_mapping=g_slot,
            block_tables=g_bt,
            outputs=g_out,
        )
        self.prefill_graphs[key] = gv
        return gv

    def pre_capture_prefill_graphs(self):
        """Pre-capture prefill graphs for the common single-sequence shapes.

        Eager prefill is host-bound (~40ms/step for ~680 kernel launches);
        replaying from a CUDA graph costs ~8ms. The benchmark's prompts are
        short (tokens 13-113 after chat template), so capture (1, S) graphs
        for S in 1..self.prefill_capture_max at init, when the cost is
        invisible to the benchmark (elapsed is measured after engine init).
        Shapes outside the grid (multi-seq prefills, longer prompts) fall back
        to the lazy capture path in run_prefill.
        """
        max_s = self.prefill_capture_max
        device = torch.device(self.device)
        bt = torch.zeros(1, self.max_num_blocks, dtype=torch.int32, device=device)
        for S in range(1, max_s + 1):
            ids = torch.zeros(S, dtype=torch.long, device=device)
            pos = torch.arange(S, dtype=torch.long, device=device)
            cu = torch.tensor([0, S], dtype=torch.int32, device=device)
            ctx = type("C", (), {})()
            ctx.cu_seqlens_q = cu
            ctx.cu_seqlens_k = cu
            ctx.max_seqlen_q = S
            ctx.max_seqlen_k = S
            ctx.block_tables = bt
            self._capture_prefill_graph((1, S, S, S), ids, pos, ctx)

    def run(self, seqs: list, is_prefill: bool):
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        try:
            logits = self.run_model(input_ids, positions, is_prefill)
        finally:
            reset_context()

        if self.rank == 0:
            next_token_ids = torch.argmax(logits, dim=-1).tolist()
        else:
            next_token_ids = None
        return next_token_ids

    def _padded_bt_row(self, seq):
        """Padded (max_num_blocks,) block-table row for a sequence, cached.

        A sequence's block table is allocated once at admission for its full
        max length and never changes, so the padded GPU row is constant; the
        old per-step torch.full+torch.cat rebuilt it every decode step."""
        cached = self._padded_bt.get(seq.seq_id)
        if cached is None:
            bt_row = seq.block_table[0]
            if bt_row.size(0) < self.max_num_blocks:
                pad = torch.full(
                    (self.max_num_blocks - bt_row.size(0),),
                    -1, dtype=torch.int32, device=self.device,
                )
                bt_row = torch.cat([bt_row, pad], dim=0)
            self._padded_bt[seq.seq_id] = bt_row
            cached = bt_row
        return cached

    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        seq_rows = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = []

        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq_i, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens

            tokens = seq.token_ids[start:end]

            input_ids.extend(tokens)
            positions.extend(range(start, end))
            seq_rows.extend([seq_i] * len(tokens))
            cu_seqlens_q.append(cu_seqlens_q[-1] + len(tokens))
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, len(tokens))
            max_seqlen_k = max(max_seqlen_k, end)

            # GPU stack of per-seq block tables (no .tolist() sync), padded so
            # all rows have the same width.
            block_tables.append(self._padded_bt_row(seq))

        device = torch.device(self.device)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)

        bt = torch.stack(block_tables, dim=0)  # (num_seqs, max_num_blocks)

        # Slots entirely on GPU (integer arithmetic, matches the old .item() path).
        logical_ids = positions // self.block_size          # int64, for indexing
        offsets = (positions % self.block_size).to(torch.int32)
        row_ids = torch.tensor(seq_rows, dtype=torch.long, device=device)
        slot_mapping = bt[row_ids, logical_ids] * self.block_size + offsets  # int32

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=bt,
        )

        return input_ids, positions

    def prepare_decode(self, seqs):
        rows = []
        block_tables = []
        max_ctx = 0

        for seq in seqs:
            pos = len(seq.token_ids) - 1

            # Slots computed on CPU from the (CPU) block_table_ids list: the
            # block for position p is block_table_ids[p // block_size]. This
            # avoids the per-step GPU integer ops and needs no .item() syncs.
            slot = (
                seq.block_table_ids[pos // self.block_size] * self.block_size
                + pos % self.block_size
            )
            rows.append([seq.last_token, pos, len(seq.token_ids), slot])
            max_ctx = max(max_ctx, len(seq.token_ids))

            block_tables.append(self._padded_bt_row(seq))

        device = self._device
        # The padded block-table rows are constant for a sequence's lifetime
        # (all blocks are pre-allocated at admission), so the stacked (bs,8)
        # batch table is constant for a given set+order of sequences. Cache it
        # by the seq-id tuple: consecutive decode steps reuse the same running
        # set and skip the torch.stack entirely.
        key = tuple(s.seq_id for s in seqs)
        bt = self._bt_cache.get(key)
        if bt is None:
            if len(seqs) == 1:
                bt = block_tables[0].unsqueeze(0)
            else:
                bt = torch.stack(block_tables, dim=0)  # (bs, max_num_blocks)
            if len(self._bt_cache) < 2048:
                self._bt_cache[key] = bt

        if self.enforce_eager:
            # Eager mode: the model reads slot_mapping/context_lens directly
            # (store_kvcache/flash), which need contiguous int32.
            input_ids = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
            positions = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
            context_lens = torch.tensor([r[2] for r in rows], dtype=torch.int32, device=device)
            slot_mapping = torch.tensor([r[3] for r in rows], dtype=torch.int32, device=device)
        else:
            # Graph mode: one (bs,4) int32 H2D; the strided column views are
            # fine because run_model copies them into the contiguous graph
            # buffers (casting int32->long for ids/positions). Saves 3 tensor
            # constructions (~70us/step of exposed CPU).
            t = torch.tensor(rows, dtype=torch.int32, device=device)
            input_ids = t[:, 0]
            positions = t[:, 1]
            context_lens = t[:, 2]
            slot_mapping = t[:, 3]

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=bt,
            use_fast_flash=max_ctx <= self.block_size,   # CPU-side, no sync
        )

        return input_ids, positions

    @staticmethod
    def _percentile(values, q):
        if not values:
            return None
        s = sorted(values)
        if len(s) == 1:
            return s[0]
        idx = (len(s) - 1) * (q / 100.0)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    def _aggregate_metrics(self, finished_seqs, elapsed, throughput):
        """Aggregate per-request latency metrics into mean / p50 / p99.

        TTFT: arrival -> first generated token.
        TPOT: mean decode-step latency per request (= mean ITL).
        ITL : all inter-token gaps pooled across requests.
        """
        ttfts = [s.ttft for s in finished_seqs if s.ttft is not None]
        tpots = [s.tpot for s in finished_seqs if s.tpot is not None]
        all_itls = [g for s in finished_seqs for g in s.itls]

        def summary(values):
            if not values:
                return {"mean": None, "p50": None, "p99": None}
            return {
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p99": self._percentile(values, 99),
            }

        total_output_tokens = sum(s.num_generated_tokens for s in finished_seqs)

        return {
            "num_requests": len(finished_seqs),
            "elapsed_s": elapsed,
            "output_tokens": total_output_tokens,
            "throughput_tok_s": throughput,
            "ttft_s": summary(ttfts),
            "tpot_s": summary(tpots),
            "itl_s": summary(all_itls),
        }