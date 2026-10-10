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

from src.scheduler import Scheduler
from utils.sequence import Sequence
from utils.sampling_params import SamplingParams
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
        self.background_task = None

        self.scheduler = Scheduler(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
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

        # Keep the tokenized ids on CPU: they are only consumed as a Python list
        # (Sequence.token_ids, block matching). Moving to GPU and back is a pure
        # round-trip that also puts a GPU sync on the event-loop submission path,
        # delaying subsequent arrivals on the benchmark.
        model_inputs = self.tokenizer([text], return_tensors="pt")
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
                # A request may have been submitted between the is_finished()
                # check and the clear(): add_async_request sets has_work after
                # scheduler.add, so clearing here would drop that wakeup and the
                # request would sit in the queue until the next arrival. Re-check
                # after clearing; only sleep if it is still empty.
                if self.scheduler.is_finished():
                    await self.has_work.wait()
                    continue
                # Work appeared in the window; fall through and step.

            try:
                # Run the (blocking, GPU-syncing) step in a worker thread so the
                # event loop stays free to service request submissions on time.
                # Without this, the ~5.5ms synchronous step starves the
                # submission coroutine and arrivals drift late by several
                # seconds over a 400-request run, inflating end-to-end latency.
                mode = os.environ.get("TOKN_STEP_MODE", "split")
                if mode == "split":
                    # Decode steps enqueue on the event loop (keeps arrivals
                    # on time) with only the GPU wait off-loaded; prefill
                    # steps run entirely in the executor (their lazy CUDA-graph
                    # capture must not block the loop). See _step_split.
                    finished = await self._step_split()
                elif mode == "inline":
                    finished = self.step()
                else:
                    finished = await asyncio.get_running_loop().run_in_executor(
                        None, self.step
                    )

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
                logger.exception("run_loop failed; failing all pending requests")
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

        return self._step_with_seqs(seqs, is_prefill)

    @torch.inference_mode
    def _step_with_seqs(self, seqs, is_prefill):
        """Full synchronous step given already-scheduled seqs: prepare, GPU
        forward, argmax, postprocess, block hashing/deallocation."""
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
            self.RUN._padded_bt.pop(seq.seq_id, None)

        return finished

    async def _step_split(self):
        """Non-blocking step: enqueue decode GPU work on the event loop, then
        await GPU completion without blocking the loop.

        The full run_in_executor step keeps arrivals on time but costs
        ~0.2ms/step of executor dispatch + GIL handoff (TPOT 3.46 vs 3.25 for
        inline). Here the decode enqueue phase (schedule/prepare/replay/argmax)
        runs directly on the event loop (~0.15ms, fine since arrivals are 2/s)
        and ONLY the GPU wait is delegated to the executor as a bare
        `event.synchronize()` (a C call, no GIL churn). PREFILL steps run the
        whole step in the executor: run_prefill can lazily CAPTURE a multi-seq
        CUDA graph (~190ms) on first use, which must not block the event loop
        (on-loop capture regressed TTFT p99 140->217ms in exp 28)."""
        loop = asyncio.get_running_loop()
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return []

        if is_prefill:
            # Prefills may trigger lazy graph capture (~190ms); keep them off
            # the event loop. Prefills are rare (~1 per request vs ~256 decode
            # steps), so the full-step executor tax is negligible here.
            return await loop.run_in_executor(None, self._step_with_seqs, seqs, True)

        with torch.inference_mode():
            input_ids, positions = self.RUN.prepare_decode(seqs)
            try:
                logits = self.RUN.run_model(input_ids, positions, False)
            finally:
                reset_context()
            # Enqueue the argmax reduction now (reads the graph's output buffer
            # on the same stream); only the completion wait is deferred.
            tokens_t = torch.argmax(logits, dim=-1)
            ev = torch.cuda.Event()
            ev.record()

        # Wait for the GPU without blocking the event loop. The executor task
        # is a single C call (no Python), so there is no GIL contention.
        await loop.run_in_executor(None, ev.synchronize)

        with torch.inference_mode():
            next_token_ids = tokens_t.tolist()
            finished = self.scheduler.postprocess(seqs, next_token_ids, False)

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
            self.RUN._padded_bt.pop(seq.seq_id, None)

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