import math
import torch
from torch import Tensor
nn = torch.nn
F = nn.functional


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class GatedMLP(nn.Module):
    def __init__(self, dim: int = 1024, expansion_factor: int = 2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = Linear(dim, 2 * hidden, bias=False)
        self.shrink = Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim ** -0.5)
            self.shrink.weight.normal_(std=hidden ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return self.shrink(x)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = Linear(dim, 4 * dim)
        self.c_proj = Linear(4 * dim, dim)
        with torch.no_grad():
            nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)