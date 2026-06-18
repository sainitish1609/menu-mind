import torch

# hyperparameters
batch_size = 16 # amount of independent sequences will we process in parallel?
block_size = 256 # context length
max_iters = 3000
eval_interval = 250
learning_rate = 3e-4
device_type = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cuda' if 'cuda' in device_type else ('mps' if device_type == 'mps' else 'cpu')
use_amp = (device == 'cuda')   # bf16 autocast is reliable on CUDA
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3