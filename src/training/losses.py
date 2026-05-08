import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Pearson R across all (channel, timepoint) pairs.
    pred, target: (N, C, T) tensors."""
    p = pred   - pred.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    num = (p * t).sum(0)
    den = (p.pow(2).sum(0) * t.pow(2).sum(0)).sqrt().clamp(min=1e-8)
    return (num / den).mean()
