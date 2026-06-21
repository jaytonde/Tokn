from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 2048
    ignore_eos: bool = False