import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from memory import memory
import math

def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )


class STU(nn.Module):
    def __init__(self, n_embd, num_eigh, torch_dtype, phi, n, idx, gating: bool = False) -> None:
        super(STU, self).__init__()
        self.phi = phi
        self.n = n
        self.K = num_eigh
        self.dim = n_embd
        self.gating = gating
        self.M_phi_plus = nn.Parameter(
            torch.empty(self.K, self.dim, self.dim, dtype=torch_dtype)
        )
        self.M_phi_minus = nn.Parameter(
            torch.empty(self.K, self.dim, self.dim, dtype=torch_dtype)
        )
        if self.gating:
            self.cross_attn = memory(n_embd, idx=idx, block_size=n)
        nparams = self.num_params() / 1e6
        print("Number of parameters: %.3fM" % (nparams,))

    def num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gating:
            x, mem = self.cross_attn(x, None)
        else:
            mem = None
        return self.spectral(x), mem

    def spectral(self, x: torch.Tensor) -> torch.Tensor:
        U_plus, U_minus = convolve(x, self.phi, self.n)
        # U_plus shape is (batch, seqlen, num_eigh(self.K), n_embd)
        # U_minus shape is (batch, seqlen, num_eigh(self.K), n_embd)
        # Then, contract over the K and d_in dimensions
        spectral_plus = torch.tensordot(
            U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
        )
        spectral_minus = torch.tensordot(
            U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
        )

        return spectral_plus + spectral_minus



def get_hankel(seq_len: int) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int, 
    K: int, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    Z = get_hankel(seq_len)
    sigma, phi = np.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma ** 0.25
    return torch.tensor(phi, device=device, dtype=dtype)



def convolve(u: torch.Tensor, v: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    _, K = v.shape
    sgn = sgn.unsqueeze(-1)
    v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, d_in, stack)
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus


phi = get_spectral_filters(16 * 1024, num_eigh=24, device='cuda', torch_dtype=torch.float32)


if __name__ == '__main__':
    seq_len = 64
    num_eigh = 24
    n_embd = 1024
    device = 'cuda'
    torch_dtype = torch.float32
    n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
    phi = get_spectral_filters(seq_len, num_eigh, device, torch_dtype)

    layer = STU(
        n_embd=n_embd,
        num_eigh=num_eigh,
        torch_dtype=torch_dtype,
        phi=phi,
        n=n,
    ).to(device)

    x = torch.randn(2, seq_len, n_embd).to(device)

    print(layer(x).shape)
