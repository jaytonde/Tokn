import re
import time
import logging
import torch
from dataclasses import dataclass
from transformers  import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

from context import reset_context, set_context
from qwen3 import Qwen3ForCausalLM
from loader import load_model


from scheduler import Scheduler
from sequence import Sequence



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Engine:

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu", dtype: str = "fp16"):
        self.model = model
        self.dtype = torch.float16 if dtype=="fp16" else torch.bfloat16
        self.device = device

        self.hf_config = AutoConfig.from_pretrained(self.model)

        self.custom_model = Qwen3ForCausalLM(self.hf_config)

        self.scheduler = None
        self.outputs = {}

    def _make_block_table(self, max_len: int, device: torch.device):
        num_blocks = (max_len + self.block_size - 1) // self.block_size

        if num_blocks > self.num_blocks:
            raise RuntimeError(
                f"Request needs {num_blocks} KV blocks, "
                f"but only {self.num_blocks} are allocated."
            )

        block_table = torch.arange(
            num_blocks,
            device=device,
            dtype=torch.int32,
        ).unsqueeze(0)  # shape: [1, num_blocks]

        return block_table

    def _make_slot_mapping(
        self,
        start_pos: int,
        num_tokens: int,
        block_table: torch.Tensor,
        device: torch.device ):
        slots = []

        for pos in range(start_pos, start_pos + num_tokens):
            logical_block_id = pos // self.block_size
            block_offset = pos % self.block_size

            physical_block_id = block_table[0, logical_block_id].item()
            slot = physical_block_id * self.block_size + block_offset

            slots.append(slot)

        return torch.tensor(slots, device=device, dtype=torch.int32)

    def _set_prefill_context(
        self,
        seq_len: int,
        block_table: torch.Tensor,
        device: torch.device):

        cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)

        slot_mapping = self._make_slot_mapping(
            start_pos=0,
            num_tokens=seq_len,
            block_table=block_table,
            device=device,
        )

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
            block_tables=None,  # no prefix cache for now
        )

    def _set_decode_context(
        self,
        pos: int,
        context_len: int,
        block_table: torch.Tensor,
        device: torch.device):
        slot_mapping = self._make_slot_mapping(
            start_pos=pos,
            num_tokens=1,
            block_table=block_table,
            device=device,
        )

        context_lens = torch.tensor(
            [context_len],
            device=device,
            dtype=torch.int32,
        )

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_table,
        )

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

        self.block_size = 256
        self.num_blocks = 256 * 2

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads

        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads

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

        layer_id = 0
        for layer in self.custom_model.model.layers:
            layer.self_attn.attn.k_cache = self.kv_cache[0, layer_id]
            layer.self_attn.attn.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1

        logger.info(
            f"KV cache allocated: "
            f"layers={num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, kv_heads={num_kv_heads}, head_dim={head_dim}"
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

        self.scheduler = Scheduler(
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=self.dtype).to(self.device)
        logger.info("HF reference model loaded")

    def custom_generate(self, prompt: str, max_tokens: int = 1024):
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # returns the formatted string instead of token IDs
            add_generation_prompt=True #appends the assistant turn header (<|im_start|>assistant\n) so the model knows to start generating a response
        ) 

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = model_inputs["input_ids"]
        print(f"generated_ids:{generated_ids}")

        positions = torch.arange(generated_ids.shape[1], device=self.model.device)
        print(f"positions:{positions}")

        with torch.inference_mode():
            t_start = time.perf_counter()
            for step in range(max_tokens):
                self._set_prefill_context(generated_ids.shape[1], generated_ids.device)
                try:
                    logits = self.custom_model(input_ids=generated_ids.flatten(), positions=positions)
                finally:
                    reset_context()
                next_token_logits = logits[-1, :].unsqueeze(0)
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                positions = torch.arange(generated_ids.shape[1], device=self.model.device)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    logger.info(f"EOS token hit at step {step + 1}")
                    break
            elapsed = time.perf_counter() - t_start

        num_generated = generated_ids.shape[1] - model_inputs["input_ids"].shape[1]
        tok_per_sec = num_generated / elapsed if elapsed > 0 else 0
        logger.info(f"Generation done: {num_generated} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

        new_tokens = generated_ids[:, model_inputs["input_ids"].shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        response = response.split("</think>")[-1]

        return response

    def generate(self, prompt: str, max_tokens: int = 1024):

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        print(f"model_inputs:{model_inputs}")

        generated_ids = model_inputs["input_ids"]
        with torch.inference_mode():
            for _ in range(max_tokens):
                outputs = self.model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        new_tokens = generated_ids[:, model_inputs["input_ids"].shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return response

    def seperate_prefill_decode(self, prompt: str, max_tokens: int = 1024):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = model_inputs["input_ids"]

        prompt_len = generated_ids.shape[1]
        max_total_len = prompt_len + max_tokens

        block_table = self._make_block_table(
            max_len=max_total_len,
            device=generated_ids.device,
        )

        with torch.inference_mode():
            t_start = time.perf_counter()

            positions = torch.arange(prompt_len, device=generated_ids.device)

            #####----------PREFILL------------#####

            self._set_prefill_context(
                seq_len=prompt_len,
                block_table=block_table,
                device=generated_ids.device,
            )

            try:
                logits = self.custom_model(
                    input_ids=generated_ids.flatten(),
                    positions=positions,
                )
            finally:
                reset_context()

            next_token_logits = logits[-1, :].unsqueeze(0)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                new_tokens = generated_ids[:, prompt_len:]
                return self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                )[0]
            

            #####----------DECODE------------#####
                
            for step in range(1, max_tokens):
                pos = generated_ids.shape[1] - 1
                context_len = generated_ids.shape[1]

                input_id = generated_ids[:, -1].flatten()
                positions = torch.tensor([pos], device=generated_ids.device)

                self._set_decode_context(
                    pos=pos,
                    context_len=context_len,
                    block_table=block_table,
                    device=generated_ids.device,
                )

                try:
                    logits = self.custom_model(
                        input_ids=input_id,
                        positions=positions,
                    )
                finally:
                    reset_context()

                next_token_logits = logits[-1, :].unsqueeze(0)
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    logger.info(f"EOS token hit at step {step + 1}")
                    break

            elapsed = time.perf_counter() - t_start

            num_generated = generated_ids.shape[1] - prompt_len
            tok_per_sec = num_generated / elapsed if elapsed > 0 else 0
            logger.info(
                f"Generation done: {num_generated} tokens in {elapsed:.2f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

            new_tokens = generated_ids[:, prompt_len:]
            response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            response = response.split("</think>")[-1]

            return response


    #Scheduler functions

    def prepare_prefill():
        pass

    def prepare_decode():
        pass

    def add_request(self, prompt: str, max_tokens: int = 128):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # returns the formatted string instead of token IDs
            add_generation_prompt=True #appends the assistant turn header (<|im_start|>assistant\n) so the model knows to start generating a response
        ) 

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = model_inputs["input_ids"]

        seq = Sequence(
            token_ids = generated_ids,
            max_tokens = max_tokens
        )

        prompt_len = generated_ids.shape[1]
        max_total_len = prompt_len + max_tokens

        seq.block_table = self._make_block_table(
            max_len=max_total_len,
            device=generated_ids.device,
        )
        
        self.scheduler.add(seq)

    @torch.inference_mode
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()

        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        try:
            logits = self.custom_model(
                input_ids=input_ids,
                positions=positions,
            )
        finally:
            reset_context()

        next_token_ids = torch.argmax(logits, dim=-1).tolist()

        finished = self.scheduler.postprocess(
            seqs=seqs,
            token_ids=next_token_ids,
            is_prefill=is_prefill,
        )

        return finished