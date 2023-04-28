import torch
assert torch.__version__ >= '1.6.0'
import torch.nn as nn
import numpy as np


def layer_norm(d_model, condition=True):
    return nn.LayerNorm(d_model) if condition else None


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_proposal_mask(T, B):
    mask = torch.zeros(T, (T + 1) * T // 2)
    for sz, idx in zip(range(1, T + 1), np.cumsum(range(T))):
        mask[:sz, idx: idx + sz] = torch.fliplr(torch.tril(torch.ones(sz, sz)))
    mask = mask.unsqueeze(1).repeat(1, B, 1)
    return mask
