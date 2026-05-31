from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 128
    ignore_eos: bool = False