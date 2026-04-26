import re
import torch
from dataclasses import dataclass
from transformers  import AutoTokenizer, AutoModelForCausalLM

from qwen3 import Qwen3ForCausalLM

class Engine:
    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu", dtype: str = "fp16"):
        self.model = model
        self.dtype = torch.float16 if dtype=="fp16" else torch.bfloat16
        self.device = device

        self.custom_model = Qwen3ForCausalLM()

    def load(self):
        print(f"Loading model onto device : {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=self.dtype).to(self.device)
        print("Model loaded successfully!!!")

    def custom_generate(self, prompt: str, max_tokens: int = 1024):
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = model_inputs["input_ids"]
        with torch.inference_mode():
            for _ in range(max_tokens):
                outputs = self.custom_model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        new_tokens = generated_ids[:, model_inputs["input_ids"].shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

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

