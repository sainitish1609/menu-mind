# menu-mind

A small, modern, decoder-only language model built from scratch in PyTorch — and the
text half of a vision language model (VLM) that is now being wired together.

The project starts from nanoGPT and upgrades it, one piece at a time, into a
contemporary Llama-style transformer: BPE tokenization, rotary position embeddings,
RMSNorm, SwiGLU, Flash Attention, weight tying, and a modern training recipe. It then
goes one step further: a pretrained vision encoder and a projector are spliced onto the
front of the model so that, with image–text data, it can learn to describe menu items
from a photo.

The model is trained on a menu dataset and learns to generate the **details** of a dish
(category, description, price) given its **name**.

## Current status

- **Language model** — complete, trained, and saving correctly.
- **Vision encoder** — built. It uses a frozen, pretrained SigLIP2 backbone plus a small
  trainable projector (`vision_encoder.py`), and the model already concatenates the
  projected image embeddings onto the front of the text sequence inside `forward`.
- **Multimodal training path** — wired up end to end (image tokens are spliced in, their
  logits are dropped before the loss, and the prompt is masked). It currently runs
  **text-only**, because the `images` list in `test.py` is empty by default. Drop image
  files in and point the list at them to start training the projector on real photos.

## Architecture

A GPT-style decoder with the following modern components:

- **Tokenizer** — tiktoken BPE (`gpt2` encoding, ~50k vocabulary), subword tokens rather than characters.
- **Positional encoding** — Rotary Position Embeddings (RoPE), applied to queries and keys inside attention. No learned absolute position table. The RoPE cache covers `block_size + 256` positions so it spans the spliced-in image tokens too.
- **Normalization** — RMSNorm (pre-norm), in place of LayerNorm.
- **Attention** — multi-head self-attention using PyTorch's fused `scaled_dot_product_attention` (Flash Attention), causal masking via `is_causal=True`.
- **Feed-forward** — SwiGLU gated MLP (`silu(W_gate · x) * (W_up · x)` then `W_down`), hidden size ~8/3 × embedding dim.
- **Weight tying** — the token embedding table and the output head share one weight matrix.
- **No biases** on the linear layers, following the Llama convention.
- **Vision encoder** — a frozen `google/siglip2-base-patch16-256` model that turns a 256×256 image into 256 patch features, cached locally under `models/siglip2/`.
- **Projector** — a single `nn.Linear` that resizes SigLIP2's 768-dim patch features to the model's embedding width (`n_embd`) so images and text share one space.

Default model size (set in `hyperparameters.py`):

| Hyperparameter | Value |
| --- | --- |
| `n_embd` (embedding width) | 384 |
| `n_layer` (transformer blocks) | 6 |
| `n_head` (attention heads) | 6 |
| `block_size` (context length) | 256 |
| `dropout` | 0.3 |

The language model is roughly **30M parameters** (this is the count printed at startup).
The frozen SigLIP2 backbone adds its own (untrained) weights on top, plus the small
trainable projector.

## Data and task format

Training data is built from a menu dataset (`datasets/Menu Items.csv`) by `test.py`,
which samples rows and formats each one as an instruction-style prompt/completion pair:

```
### Item: <name>
### Details:
Category: <category>
Description: <description>
Price: <price>
<|endoftext|>
```

The text up to and including `### Details:\n` is treated as the **prompt**; everything
after it (ending in the `<|endoftext|>` stop token) is the **answer**. During training
the prompt region is masked out of the loss, so the model is scored only on the details
it is supposed to generate.

## Training recipe

- **Optimizer** — AdamW with `betas=(0.9, 0.95)`.
- **Weight decay** — 0.1 on 2D weight matrices and embeddings; 0 on 1D parameters (norms).
- **Learning rate** — linear warmup then cosine decay (see `get_lr` in `helpers.py`).
- **Gradient clipping** — global norm clipped to 1.0.
- **Mixed precision** — bfloat16 autocast on CUDA (disabled on MPS/CPU via `use_amp`).
- **Loss** — next-token cross-entropy, with the **prompt** region masked (`-100`),
  **padding** positions masked, and any **image-token** logits removed before scoring.

## Project layout

```
menu-mind/
├── main.py             # model definition + training loop (main entry point)
├── generate.py         # interactive inference: load weights, prompt the model
├── vision_encoder.py   # frozen SigLIP2 encoder + projector
├── helpers.py          # get_lr, rotate_half, apply_rope
├── hyperparameters.py  # all hyperparameters + device / use_amp setup
├── test.py             # dataset builder (CSV → input.txt) + the images list
├── datasets/
│   └── Menu Items.csv  # source data — not committed
├── models/siglip2/     # locally cached SigLIP2 weights — not committed
├── input.txt           # generated training data — not committed
├── menu_model.pth      # saved model weights (produced after training) — not committed
└── README.md
```

## Setup

Requires Python 3.10+ and PyTorch.

```bash
pip install torch tiktoken transformers pillow pandas
```

Build the training data from the menu CSV (writes `input.txt`):

```bash
python test.py
```

The first run also downloads the SigLIP2 weights from Hugging Face and caches them under
`models/siglip2/` for offline reuse.

## Usage

Train the model and generate a sample:

```bash
python main.py
```

This prints the parameter count, logs train/val loss during training, generates 200
tokens of sample details for a fixed example item at the end, and saves the trained
weights to `menu_model.pth`.

Then chat with the trained model interactively:

```bash
python generate.py
```

You type an item name, it wraps it in the trained `### Item: … ### Details:` format and
generates the details. Enter `quit` to exit.

## Notes on hardware and memory

The large BPE vocabulary makes the output projection and its logits tensor the dominant
memory cost. The logits are shaped `(batch_size, sequence_length, vocab_size)`, so memory
scales directly with `batch_size`. If training runs out of memory, reduce `batch_size`
first, then `block_size` if needed.

bfloat16 autocast only accelerates training on CUDA GPUs. On Apple Silicon (MPS) the
model still runs on the GPU but in float32; `use_amp` is set to `False` there so the
autocast block becomes a safe no-op.

## Roadmap: from LM to VLM

The language model and the plumbing for vision are both in place. Components:

1. **Vision encoder** — ✅ pretrained SigLIP2 turns an image into a sequence of patch embeddings.
2. **Projector** — ✅ a small linear layer resizes the image embeddings to `n_embd` so they live in the same space as text tokens.
3. **Splice** — ✅ the projected image embeddings are concatenated onto the front of the token embedding sequence, and the transformer runs over the combined row.
4. **Training** — ⏳ the loss is already masked so only the text answer is scored; what remains is supplying real image–text pairs (populate the `images` list / pair photos with menu rows) and training the projector on them.

## Acknowledgements

Built on the foundation of nanoGPT. Architectural choices follow the
Llama / modern decoder-only transformer family. The vision encoder uses Google's SigLIP2.