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
        self.graph_pool = None
        self.graph_bs = []
        self.graph_vars = {}

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
        outputs        = torch.zeros(max_bs, hidden_size, dtype=self.dtype, device=device)

        self.graph_bs  = [bs for bs in (1,2,3,4) if bs <= max_bs]

        if max_bs not in self.graph_bs:
            self.graph_bs.append(max_bs)


        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs]
            )

            # Warmup run (lazy allocs / autotune happen OUTSIDE the graph).
            outputs[:bs] = self.custom_model.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.custom_model.model(input_ids[:bs], positions[:bs])

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
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

        if is_prefill or self.enforce_eager or not self.graphs or input_ids.size(0) > self.graph_bs[-1]:
            return self.custom_model(input_ids, positions)

        bs      = input_ids.size(0)
        context = get_context()
        graph   = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        gv      = self.graph_vars

        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)          # padding rows write nowhere
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].zero_()            # padding rows attend to nothing
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        graph.replay()

        return self.custom_model.compute_logits(gv["outputs"][:bs])

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

    def prepare_prefill(self, seqs):
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = []

        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens

            tokens = seq.token_ids[start:end]

            input_ids.extend(tokens)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + len(tokens))
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, len(tokens))
            max_seqlen_k = max(max_seqlen_k, end)

            for pos in range(start, end):
                logical_block_id = pos // self.block_size
                block_offset = pos % self.block_size
                physical_block_id = seq.block_table[0, logical_block_id].item()
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

            block_tables.append(seq.block_table.squeeze(0).tolist())

        max_blocks = max(len(x) for x in block_tables)
        block_tables = [x + [-1] * (max_blocks - len(x)) for x in block_tables]

        device = torch.device(self.device)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )

        return input_ids, positions

    def prepare_decode(self, seqs):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        block_tables = []

        for seq in seqs:
            pos = len(seq.token_ids) - 1

            input_ids.append(seq.last_token)
            positions.append(pos)
            context_lens.append(len(seq.token_ids))

            logical_block_id = pos // self.block_size
            block_offset = pos % self.block_size
            physical_block_id = seq.block_table[0, logical_block_id].item()
            slot = physical_block_id * self.block_size + block_offset
            slot_mapping.append(slot)

            block_tables.append(seq.block_table.squeeze(0).tolist())

        max_blocks = max(len(x) for x in block_tables)

        block_tables = [x + [-1] * (max_blocks - len(x)) for x in block_tables] #wht?

        device = torch.device(self.device)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, device=device)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
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