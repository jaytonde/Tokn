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
import torch.multiprocessing as mp

from src.run import Run as RUN



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Engine:

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", enforce_eager=False, device: str = "cpu", dtype: str = "fp16", max_model_len: int = 2048, config: dict = None):
        self.model = model
        self.dtype = torch.float16 if dtype=="fp16" else torch.bfloat16
        self.device = device
        self.hf_config = AutoConfig.from_pretrained(self.model)


        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=RUN, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.RUN = RUN(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        self.block_manager = self.RUN.block_manager
        self.block_size = self.RUN.block_size

        self.request_states = {}
        self.request_lock = asyncio.Lock()
        self.has_work = asyncio.Event()

        self.scheduler = Scheduler(
            max_num_seqs=5,
            max_num_batched_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
            block_manager=self.block_manager,
            device=self.device,
        )
 
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

    @torch.inference_mode
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return []

        next_token_ids = self.RUN.call("run", seqs, is_prefill)
        
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