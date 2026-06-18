# menu-mind

A small, modern, decoder-only language model built from scratch in PyTorch — and the
language-model half of a vision language model (VLM) in progress.

The project starts from nanoGPT and upgrades it,
one piece at a time, into a contemporary Llama-style transformer: BPE tokenization,
rotary position embeddings, RMSNorm, SwiGLU, Flash Attention, weight tying, and a
modern training recipe. The eventual goal is to add a vision encoder and turn it into
a VLM that can answer questions about images.

## Current status

The language model is complete, trained, and saving correctly. The vision encoder
(the piece that turns an image into embeddings the model can read) is the next step
and is **not yet built**.

## Architecture

A GPT-style decoder with the following modern components:

- **Tokenizer** — tiktoken BPE (`gpt2` encoding, ~50k vocabulary), subword tokens rather than characters.
- **Positional encoding** — Rotary Position Embeddings (RoPE), applied to queries and keys inside attention. No learned absolute position table.
- **Normalization** — RMSNorm (pre-norm), in place of LayerNorm.
- **Attention** — multi-head self-attention using PyTorch's fused `scaled_dot_product_attention` (Flash Attention), causal masking via `is_causal=True`.
- **Feed-forward** — SwiGLU gated MLP (`silu(W_gate · x) * (W_up · x)` then `W_down`), hidden size ~8/3 × embedding dim.
- **Weight tying** — the token embedding table and the output head share one weight matrix.
- **No biases** on the linear layers, following the Llama convention.

Default model size (set in `hyperparameters.py`):

| Hyperparameter | Value |
| --- | --- |
| `n_embd` (embedding width) | 384 |
| `n_layer` (transformer blocks) | 6 |
| `n_head` (attention heads) | 6 |
| `block_size` (context length) | 256 |
| `dropout` | 0.2 |

This produces roughly **30M parameters**.

## Training recipe

- **Optimizer** — AdamW with `betas=(0.9, 0.95)`.
- **Weight decay** — 0.1 on 2D weight matrices and embeddings; 0 on 1D parameters (norms).
- **Learning rate** — linear warmup then cosine decay (see `get_lr` in `helpers.py`).
- **Gradient clipping** — global norm clipped to 1.0.
- **Mixed precision** — bfloat16 autocast on CUDA (disabled on MPS/CPU via `use_amp`).
- **Loss** — next-token cross-entropy.

## Project layout

```
menu-mind/
├── algorithm.py        # model definition + training loop (main entry point)
├── helpers.py          # get_lr, rotate_half, apply_rope
├── hyperparameters.py  # all hyperparameters + device / use_amp setup
├── input.txt           # training data (e.g. Tiny Shakespeare) — not committed
├── gpt_model.pth       # saved model weights (produced after training) — not committed
└── README.md
```

## Setup

Requires Python 3.10+ and PyTorch.

```bash
pip install torch tiktoken
```

Download the training data (Tiny Shakespeare):

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

Train the model and generate a sample:

```bash
python algorithm.py
```

This prints the parameter count, logs train/val loss during training, generates 500
tokens of sample text at the end, and saves the trained weights to `gpt_model.pth`.

## Notes on hardware and memory

The large BPE vocabulary makes the output projection and its logits tensor the dominant
memory cost. The logits are shaped `(batch_size, block_size, vocab_size)`, so memory
scales directly with `batch_size`. If training runs out of memory, reduce `batch_size`
first (e.g. 64 → 16), then `block_size` if needed.

bfloat16 autocast only accelerates training on CUDA GPUs. On Apple Silicon (MPS) the
model still runs on the GPU but in float32; `use_amp` is set to `False` there so the
autocast block becomes a safe no-op.

## Roadmap: from LM to VLM

The language model is the right-hand half of a vision language model. To complete the VLM:

1. **Vision encoder** — turn an image into a sequence of patch embeddings (a from-scratch ViT, or a pretrained CLIP/SigLIP encoder).
2. **Projector** — a small layer that resizes the image embeddings to the model's embedding width (`n_embd`) so they live in the same space as text tokens.
3. **Splice** — concatenate the projected image embeddings onto the front of the token embedding sequence, then run the existing transformer over the combined row.
4. **Training** — image–text data, with the loss masked so only the text answer is scored.

## Acknowledgements

Built on the foundation of nanoGPT. Architectural choices follow the
Llama / modern decoder-only transformer family.