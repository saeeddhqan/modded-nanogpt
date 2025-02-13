from util import *
import math
import torch
nn = torch.nn
F = nn.functional

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        self.qkv_w = nn.Parameter(torch.empty(3, dim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = Linear(dim, dim)
        with torch.no_grad():
            nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)


    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
        q, k, v = qkv.view(B, T, 3 * self.num_heads, -1).chunk(3, dim=-2)
        q = self.rotary(q)
        k = self.rotary(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
