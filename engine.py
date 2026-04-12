
import torch
import time
import threading
from queue import Queue
from collections.abc import Callable, Iterator
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
            return logits.argmax().item()
        
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

            return torch.multinomial(probs, 1).item()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start_time = time.perf_counter() #Highest resolution timer available on the system (nanosecond precision on most platforms)
        first_token_time = None

        model_fn = self.model_fn or self._dummy_model
        output_tokens = []

        tokens = torch.tensor([request.prompt_tokens])

        for i in range(request.max_tokens):

            logits = model_fn(tokens)

            next_token = self._sample_token(
                logits[0, -1] if logits.dim() == 3 else logits[0],
                request.temperature,
                request.top_p
            )

            if first_token_time is None:
                first_token_time = time.perf_counter()

            output_tokens.append(next_token)

            if next_token in request.stop_token_ids:
                finish_reason = "stop"
                break

            tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)

        end_time = time.perf_counter()

        result = GenerationRequest(
            request_id=request.request_id,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            prompt_tokens=len(request.prompt_tokens),
            completion_tokens=len(output_tokens),
            time_to_first_token_ms=(first_token_time - start_time) * 1000 if first_token_time else 0,
            total_time_ms=(end_time - start_time) * 1000
        )

        with self.lock:
            #Only one thread updates the states at a time so we can avoid race condition.
            self.total_requests += 1
            self.total_tokens_generated += len(output_tokens)
            self.total_time_ms += request.total_time_ms

        return result

    def generate_stream(self, request: GenerationRequest) -> Iterator[int]:
        model_fn = self.model_fn or self._dummy_model

        tokens = torch.tensor([request.prompt_tokens])

        for i in range(request.max_tokens):
            logits = model_fn(tokens)

            next_token = self._sample_token(
                logits[0, -1] if logits.dim() == 3 else logits[0], #logits[0, -1] → takes batch 0, last sequence position → shape [32000]
                request.temperature,
                request.top_p,
            )

            yield next_token

            if next_token in request.stop_token_ids:
                break

            tokens = torch.cat([
                tokens,
                torch.tensor([[next_token]])
            ], dim=1)

    def get_stats(self) -> dict:
        with self.lock:
            avg_time = self.total_time_ms / self.total_requests if self.total_requests > 0 else 0
            throughput = self.total_tokens_generated / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0
    
            return {
                "total_requests" : self.total_requests,
                "total_tokens" : self.total_tokens_generated,
                "total_time_ms" : self.total_time_ms,
                "avg_latency_ms" : avg_time,
                "thorughtput_tokens_per_sec" : throughput
            }