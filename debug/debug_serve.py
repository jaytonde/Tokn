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
from qwen3 import Qwen3ForCausalLM
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
from qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


from tqdm import tqdm

from src.scheduler import Scheduler
from utils.sequence import Sequence
from sampling_params import SamplingParams
from src.block_manager import BlockManager
from utils.request_state import RequestState
from utils.context import get_context, reset_context, set_context



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Engine:

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", enforce_eager=False, device: str = "cpu", dtype: str = "fp16", max_model_len: int = 2048):
        self.model = model
        self.dtype = torch.float16 if dtype=="fp16" else torch.bfloat16
        self.device = device

        if device.startswith("cuda") and ":" in device:
            torch.cuda.set_device(torch.device(device))

        self.hf_config = AutoConfig.from_pretrained(self.model)

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29501")
            dist.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo", world_size=1, rank=0)

        self.custom_model = Qwen3ForCausalLM(self.hf_config)

        self.scheduler = None
        self.outputs = {}
        self.max_model_len = max_model_len
        self.metrics = None

        #for continuous batching
        self.request_states = {}
        self.request_lock = asyncio.Lock()
        self.has_work = asyncio.Event()
        self.background_task = None

        self.enforce_eager = enforce_eager
        self.graphs = {}
        self.graph_pool = None
        self.graph_bs = []
        self.graph_vars = {}

        self.load()

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
        num_kv_heads = config.num_key_value_heads

        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads

        requested_num_blocks = (self.max_model_len + self.block_size - 1) // self.block_size

        # Static sizing with headroom for multi-request batching.
        self.num_blocks = max(1024, requested_num_blocks + 16)

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

    def load(self):

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

        self.scheduler = Scheduler(
            max_num_seqs=5,
            max_num_batched_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
            block_manager=self.block_manager,
            device=self.device,
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=self.dtype).to(self.device)
        logger.info("HF reference model loaded")
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config         = self.hf_config
        hidden_size    = config.hidden_size
        max_bs         = self.scheduler.max_num_seqs
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
                is_prefill=True,
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
        graph   = self.graphs[next(x for x in self.graphs_bs if x >= bs)]
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

    async def add_async_request(self, prompt: str, sampling_params: SamplingParams | None = None) -> int:
        if sampling_params is None:
            sampling_params = SamplingParams()

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_ids_tensor = model_inputs["input_ids"][0]
        input_ids = input_ids_tensor.tolist()

        seq = Sequence(
            token_ids=input_ids,
            max_tokens=sampling_params.max_tokens,
        )

        seq.sampling_params = sampling_params
        seq.arrival_time = time.perf_counter()

        state = RequestState(
            seq=seq,
            done=asyncio.Event(),
        )

        async with self.request_lock:
            self.request_states[seq.seq_id] = state
            self.scheduler.add(seq)
            self.has_work.set()

        return seq.seq_id

    async def wait_for_result(self, seq_id: int) -> str:
        async with self.request_lock:
            state = self.request_states[seq_id]

        await state.done.wait()

        async with self.request_lock:
            state = self.request_states.pop(seq_id, state)

        if state.error is not None:
            raise state.error
        
        return state.output or ""

    async def generate_async(self, prompt: str, sampling_params: SamplingParams | None = None) -> str:
        seq_id = await self.add_async_request(prompt, sampling_params)
        return await self.wait_for_result(seq_id)

    async def run_loop(self):
        while True:
            if self.scheduler.is_finished():
                self.has_work.clear()
                await self.has_work.wait()
                continue

            try:
                finished = self.step()

                for seq in finished:
                    output_ids = seq.token_ids[seq.num_prompt_tokens:]

                    text = self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                    )
                    text = text.split("</think>")[-1]

                    async with self.request_lock:
                        state = self.request_states.get(seq.seq_id)
                        if state is not None:
                            state.output = text
                            state.done.set()

            except Exception as exc:
                async with self.request_lock:
                    for state in self.request_states.values():
                        state.error = exc
                        state.done.set()

                raise

            await asyncio.sleep(0)

    def add_request(self, prompt: str, sampling_params: SamplingParams):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # returns the formatted string instead of token IDs
            add_generation_prompt=True #appends the assistant turn header (<|im_start|>assistant\n) so the model knows to start generating a response
        ) 

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_ids_tensor = model_inputs["input_ids"][0]
        generated_ids = input_ids_tensor.tolist()

        seq = Sequence(
            token_ids = generated_ids,
            max_tokens = sampling_params.max_tokens
        )

        seq.sampling_params = sampling_params
        seq.arrival_time = time.perf_counter()

        self.scheduler.add(seq)

    @torch.inference_mode
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return []

        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        try:
            logits = self.run_model(input_ids, positions, is_prefill)
        finally:
            reset_context()

        next_token_ids = torch.argmax(logits, dim=-1).tolist()

        finished = self.scheduler.postprocess(
            seqs=seqs,
            token_ids=next_token_ids,
            is_prefill=is_prefill,
        )

        for seq in seqs:
            start_block = seq.num_hashed_blocks
            end_block = seq.num_cached_tokens // self.block_size

            if end_block > start_block:
                self.block_manager.hash_completed_blocks(
                    token_ids=seq.token_ids,
                    block_table=seq.block_table_ids,
                    start_block=start_block,
                    end_block=end_block
                )
                seq.num_hashed_blocks = end_block

        for seq in finished:
            self.block_manager.deallocate(seq.block_table_ids)

        return finished

    def generate(self, prompts: list[str], sampling_params : list[SamplingParams] = []):

        if self.background_task is not None:
            raise RuntimeError(
                "Engine.generate() cannot be used while the online background loop is running. "
                "Use generate_async() for continuous batching."
            )

        batch_arrival = time.perf_counter()

        for idx, prompt in enumerate(prompts):
            if len(sampling_params):
                self.add_request(prompt, sampling_params[idx])
            else:
                self.add_request(prompt, SamplingParams())

        for seq in self.scheduler.waiting:
            seq.arrival_time = batch_arrival

        outputs = {}

        t_start = time.perf_counter()

        finished_seqs = []
        pbar = tqdm(total=len(prompts), desc="Processing prompts")

        while not self.scheduler.is_finished():
            finished = self.step()

            for seq in finished:

                finished_seqs.append(seq)
                output_ids = seq.token_ids[seq.num_prompt_tokens:]
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs[seq.seq_id] = text.split("</think>")[-1]
                pbar.update(1)

        pbar.close()

        elapsed = time.perf_counter() - t_start

        total_output_tokens = sum(seq.num_generated_tokens for seq in finished_seqs)
        throughput = total_output_tokens / elapsed if elapsed > 0 else 0

        self.metrics = self._aggregate_metrics(finished_seqs, elapsed, throughput)

        return outputs, throughput

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