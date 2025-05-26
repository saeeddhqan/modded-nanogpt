from util import *
import math
import torch
from memory import memory
nn = torch.nn
F = nn.functional

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, idx: int, seqlen: int, is_causal: bool, num_slots: int = None, use_gating: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_attn = Linear(dim, dim * 3)
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = Linear(dim, dim)
        self.dim = dim
        self.use_gating = use_gating
        if use_gating:
            self.cross_attn = memory(
                dim,
                idx=idx,
                block_size=seqlen,
                num_slots=num_slots if num_slots is not None else math.sqrt(seqlen),
                is_causal=is_causal,
            )
            self.gate = Linear(dim, dim, bias=False)
            self.write_matter = nn.Parameter(torch.ones(dim) * 0.1)
            self.wsum = nn.Parameter(torch.tensor([1.0, -1.0]))
        # with torch.no_grad():
        #     nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        self.is_causal = is_causal


    def forward(self, x: Tensor, mem: Tensor | None) -> Tensor:
        if self.use_gating:
            y, mem = self.cross_attn(x, mem)
            w1, w2 = F.softmax(self.wsum, dim=0).split(1)
            x = w1 * x + w2 * y * F.sigmoid(self.gate(x))
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.dim, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, mem