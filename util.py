import math
import torch
from torch import Tensor
nn = torch.nn
F = nn.functional
import matplotlib.pyplot as plt


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

class SquishSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        silu = F.gelu(x)
        return silu
        # return torch.where(x < 0, silu, silu ** 2)

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = Linear(dim, 4 * dim)
        self.c_proj = Linear(4 * dim, dim)
        self.act = SquishSiLU()
        # with torch.no_grad():
            # nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
            # nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.01)
            # self.c_proj.weight.detach().zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        # x = F.relu(x).square()
        x = self.c_proj(x)
        return x


def norm1(x):
    return F.rms_norm(x, (x.size(-1),))

def norm2(x):
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
    return x / rms

if hasattr(F, 'rms_norm'):
    norm = norm1
else:
    norm = norm2

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )


class TensorTracker:
    def __init__(self):
        self.data = {}  # {name: {"means": [...], "stds": [...], "mins": [...], "maxs": [...], "hist": [...]}}

    def track(self, name, tensor: torch.Tensor):
        tensor = tensor.detach().cpu().float()
        # Basic checks
        if torch.isnan(tensor).any():
            print(f"[NaN] in {name}")
            return
        if torch.isinf(tensor).any():
            print(f"[Inf] in {name}")
            return

        if name not in self.data:
            self.data[name] = {"means": [], "stds": [], "mins": [], "maxs": [], "hist": []}

        self.data[name]["means"].append(tensor.mean().item())
        self.data[name]["stds"].append(tensor.std().item())
        self.data[name]["mins"].append(tensor.min().item())
        self.data[name]["maxs"].append(tensor.max().item())
        self.data[name]["hist"].append(tensor.flatten().numpy())


    def plot_all(self):
        for name, stats in self.data.items():
            self._plot_stats(name, stats)

    def _plot_stats(self, name, stats):
        steps = range(len(stats["means"]))
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Tensor: {name}', fontsize=16)

        axs[0].plot(steps, stats["means"], label='Mean')
        axs[0].plot(steps, stats["stds"], label='Std')
        axs[0].set_title("Mean & Std")
        axs[0].legend()

        axs[1].plot(steps, stats["mins"], label='Min')
        axs[1].plot(steps, stats["maxs"], label='Max')
        axs[1].set_title("Min & Max")
        axs[1].legend()

        axs[2].hist(stats["hist"][-1], bins=50)
        axs[2].set_title("Last Histogram")

        plt.tight_layout()
        plt.show()

def generate_synthetic_data(seqlen, dim, nsamples: int = 15):
    x_data = torch.randn(nsamples, 1, seqlen, dim)
    y_data = torch.randint(128, (nsamples, 1, seqlen))  # dummy targets
    return list(zip(x_data, y_data))