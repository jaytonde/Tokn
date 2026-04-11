
import torch
import threading
from queue import Queue
from dataclasses import dataclass, field

@dataclass
class EngineConfig:
    max_batch_size: int = 32
    max_seq_len: int = 4096
    max_total_tokens: int = 32768
    vocab_size: int = 32000
    hidden_dim: int = 4096


@dataclass
class GenerationRequest:
    request_id: int
    prompt_tokens: list[int]
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class GenerationResult:
    request_id: int
    output_tokens: list[str]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token: float
    total_time_ms: float

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.completion_tokens / (self.total_time_ms / 1000)
    

class InferenceEngine:
    def __init__(self,
                 config: EngineConfig,
                 model_fn: callable | None = None):
        
        self.config = config
        self.model_fn = model_fn

        self.pending_queue: Queue = Queue()
        self.results: dict[int, GenerationResult] = {}
        self.next_request_id = 0
        self.lock = threading.Lock()

        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_time_ms = 0.0

    def _dummy_model(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        logits     = torch.randn(batch_size, self.config.vocab_size)
        return logits

    def submit_request(self,
                       prompt_tokens: list[int],
                       max_tokens: int = 2048,
                       temperature: float = 1.0,
                       top_p: float = 1.0,
                       stop_token_ids: list[int] | None = None) -> int:
        
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1

        request = GenerationRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids or []
        )

        self.pending_queue.put(request) #add item in queue

        return request_id

    def _sample_token(self,
                      logits: torch.Tensor,
                      temperature: float = 1.0,
                      top_p: float = 1.0) -> int:
        if temperature == 0:
            return logits.softmax(logits / temperature, dim=-1)
        
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

            return torch.multinomial(probs, 1).item()

    def generate(self):
        pass

    def generate_stream(self):
        pass

    def get_stats(self):
        pass
    