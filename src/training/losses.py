import torch
import torch.nn as nn


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def pearson_correlation(pred, target):
    pass
