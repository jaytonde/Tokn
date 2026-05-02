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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class Engine:
    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu", dtype: str = "fp16"):
        self.model = model
        self.dtype = torch.float16 if dtype=="fp16" else torch.bfloat16
        self.device = device

        self.hf_config = AutoConfig.from_pretrained(self.model)

        self.custom_model = Qwen3ForCausalLM(self.hf_config)

    def _set_prefill_context(self, seq_len: int, device: torch.device):
        cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

    def load(self):

        print(f"Loading model onto device : {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        model_path = snapshot_download(self.model)
        logger.info(f"Model downloaded to: {model_path}")

        load_model(self.custom_model, model_path)
        self.custom_model = self.custom_model.to(dtype=self.dtype, device=self.device)
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
