"""Ordered sampling without replacement with exact conditional log likelihood."""
from dataclasses import dataclass
import torch

@dataclass
class Selection:
    indices: list
    log_prob: torch.Tensor
    entropy: torch.Tensor
    conditional_probs: list

def sample_ordered(probs, count, stop_index, forced=None):
    weights = probs.reshape(-1)
    if count < 1 or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid routing probabilities/capacity")
    available = int((weights[:stop_index] > 0).sum().item())
    count = min(count, available)
    selected, conditionals, logs, entropies = [], [], [], []
    for slot in range(max(count, 1)):
        mask = torch.ones_like(weights, dtype=torch.bool)
        if selected:
            mask[selected] = False
            mask[stop_index] = False
        conditional = weights.masked_fill(~mask, 0)
        if conditional.sum() <= 0:
            raise ValueError("No available router action")
        conditional = conditional / conditional.sum()
        dist = torch.distributions.Categorical(conditional)
        index = int(forced[slot]) if forced is not None else int(dist.sample().item())
        if not 0 <= index < len(weights) or conditional[index] <= 0:
            raise ValueError("Forced action is unavailable or repeated")
        selected.append(index)
        logs.append(torch.log(conditional[index]))
        entropies.append(dist.entropy())
        conditionals.append(conditional.detach().cpu().tolist())
        if index == stop_index:
            break
    if forced is not None and len(forced) != len(selected):
        raise ValueError("Forced outcome has wrong length")
    return Selection(selected, torch.stack(logs).sum(), torch.stack(entropies).sum(), conditionals)
