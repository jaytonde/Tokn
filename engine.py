from dataclasses import dataclass
from transformers  import AutoTokenizer, AutoModelForCausalLM

class Engine:
    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", dtype: str = "fp16"):
        self.model = model
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = AutoModelForCausalLM.from_pretrained(self.model)

    def generate(self, prompt: str):
        

        messages = [
            {"role": "user",
             "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=100
        )

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

