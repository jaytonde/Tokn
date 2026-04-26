import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.context as dist
from tokn.context import get_context

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank                    = dist.get_rank()
        self.tp_size                    = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings             = num_embeddings
        self.num_embeddings_per_partion = self.num_embeddings // self.tp_size
        self.vocab_start_idx            = self.num_embeddings_per_partion * self.tp_rank
        self.weight                     = nn.Parameter(torch.empty(self.num_embeddings_per_partion, embedding_dim))
        self.weight.weight_loader       = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) #Extracts only the chunk for this rank from the full loaded weight tensor.
        param_data.copy_(loaded_weight) #Copies that shard into the local parameter.

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embeddings(x, self.weight)

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y

class ParallelLMHead(VocabParallelEmbedding):
    """
    this module computes next-token logits from hidden states, 
    optimizes prefill by keeping only last-token hidden states, 
    and reconstructs full vocabulary logits from tensor-parallel shards on rank 0
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()

        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            #with tp_size > 1: full logits only on rank 0, None on other ranks

            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits

