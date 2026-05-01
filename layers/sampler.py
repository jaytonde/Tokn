import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures) #divides by temperature
        probs = torch.softmax(logits, dim=-1) #converts logits into probabilities over the vocabulary
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
    

"""
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

probs = torch.tensor([[0.70, 0.20, 0.10]])

1) torch.empty_like(probs).exponential_(1) : create exponential noise
tensor([[0.50, 2.00, 0.25]])

2) clamp_min_(1e-10) : makes sure no value is too close to zero
noise = tensor([[0.50, 2.00, 0.25]])

3) Divide probabilities by the noise
probs.div_(noise)

[[0.70/0.50, 0.20/2.00, 0.10/0.25]] = [[1.40, 0.10, 0.40]]
"""