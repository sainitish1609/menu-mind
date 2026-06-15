import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from helpers import get_lr, apply_rope
from hyperparameters import *
from datetime import datetime
from vision_encoder import SigLIP2VisionEncoder
from test import images

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab
encode = lambda s: enc.encode(s) # encoder: take a string, output a  list of integers
decode = lambda l: enc.decode(l) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x, cos, sin):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # perform the weighted aggregation of the values
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=dropout if self.training else 0.0
        )
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a linear layers followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(n_embd * 8 / 3)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = self.w_down(gate * up)
        return self.dropout(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x

class MenuLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

        head_size = n_embd // n_head
        theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        pos = torch.arange(block_size).float()
        freqs = torch.outer(pos, theta)              # (block_size, head_size/2)
        emb = torch.cat((freqs, freqs), dim=-1)      # (block_size, head_size)
        self.register_buffer('rope_cos', emb.cos())
        self.register_buffer('rope_sin', emb.sin())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, image_embeds=None):
        B, T = idx.shape

        text_embeds = self.token_embedding_table(idx) # (Batch, Time, Channels)

        if image_embeds is not None:
            x = torch.cat([image_embeds, text_embeds], dim=1)
        else:
            x = text_embeds

        T_total = x.shape[1]

        cos = self.rope_cos[:T_total]
        sin = self.rope_sin[:T_total]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            if image_embeds is not None:
                # remove visual token logits before calculating the loss
                logits_for_loss = logits[:, image_embeds.shape[1]: , :]
            else:
                logits_for_loss = logits

            B, T, C = logits_for_loss.shape
            logits_for_loss = logits_for_loss.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_for_loss, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for _ in range(max_new_tokens):
        
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    model = MenuLanguageModel()
    m = model.to(device)
    vision_encoder = SigLIP2VisionEncoder(freeze=True).to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    trainable_params = list(model.parameters()) + [
        p for p in vision_encoder.parameters() if p.requires_grad
    ]

    decay_params   = [p for p in trainable_params if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for p in trainable_params if p.dim() <  2 and p.requires_grad]

    optim_groups = [
        {'params': decay_params,   'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

    for iter in range(max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}, datetime {datetime.now().strftime('%m/%d/%Y %H:%M:%S.%f')}")
            
        # sample a batch of data
        xb, yb = get_batch('train')

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            image_embeds = vision_encoder(images)
            # evaluate the loss
            logits, loss = model(xb, yb, image_embes=image_embeds)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    torch.save(model.state_dict(), 'menu_model.pth')
    print("Model saved!")