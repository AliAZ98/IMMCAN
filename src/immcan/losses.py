"""Ranking and classification objectives."""
import torch
import torch.nn.functional as F


def listnet_loss(scores: torch.Tensor, ranks: torch.Tensor, temperature=1.0):
    """Higher predicted score is better; ground truth rank 0 is best."""
    if scores.shape != ranks.shape:
        raise ValueError(f"Shape mismatch: {scores.shape} != {ranks.shape}")
    target = F.softmax(-ranks.float() / temperature, dim=-1)
    return -(target * F.log_softmax(scores / temperature, dim=-1)).sum(-1).mean()


def smooth_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, smoothing=0.2):
    """Legacy label smoothing: mass 1-epsilon at label, epsilon/(C-1) elsewhere."""
    classes = logits.shape[-1]
    if not 0 <= smoothing < 1 or classes < 2:
        raise ValueError("Require 0 <= smoothing < 1 and at least two classes")
    log_probs = F.log_softmax(logits, dim=-1)
    target = torch.full_like(log_probs, smoothing / (classes - 1))
    target.scatter_(1, labels.reshape(-1, 1), 1 - smoothing)
    return -(target * log_probs).sum(-1).mean()
