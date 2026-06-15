import torch
from hyperparameters import max_iters, learning_rate

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # x: (B, T, head_size); cos/sin: (T, head_size)
    return x * cos + rotate_half(x) * sin

import math

warmup_iters = 100          # ramp-up length; ~2-5% of max_iters is typical
lr_decay_iters = max_iters  # decay across the whole run
min_lr = learning_rate / 10 # floor; common choice is 10% of peak

def get_lr(it):
    # 1) linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) after decay window, stay at the floor
    if it > lr_decay_iters:
        return min_lr
    # 3) cosine decay from learning_rate down to min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # goes 1 → 0
    return min_lr + coeff * (learning_rate - min_lr)