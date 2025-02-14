from util import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from memory import memory
import math
import os
try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False

K = 16

class STU(nn.Module):
    def __init__(self, n_embd, torch_dtype, phi, n, idx, K: int = K, gating: bool = False) -> None:
        super(STU, self).__init__()
        self.phi = phi
        self.n = n
        self.K = K
        self.dim = n_embd
        self.gating = gating
        self.use_approx = False
        self.flash_fft = (
            FlashFFTConv(self.n, dtype=torch.bfloat16) if
            flash_fft_available
            else None
        )

        self.M_phi_plus = nn.Parameter(
            torch.randn(self.K, self.dim, self.dim, dtype=torch_dtype) * 1e-5
        )
        self.M_phi_minus = nn.Parameter(
            torch.randn(self.K, self.dim, self.dim, dtype=torch_dtype) * 1e-5
        )
        if self.gating:
            self.cross_attn = memory(n_embd, idx=idx, block_size=n)
            self.write_matter = nn.Parameter(torch.ones(n_embd) * 0.01)

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None) -> torch.Tensor:
        if self.gating:
            y, mem = self.cross_attn(x, mem)
            x = (x + (self.write_matter * y)).to(x.dtype)
        return self.spectral(x), mem

    def spectral(self, x: torch.Tensor) -> torch.Tensor:
        # Convolve inputs and filters,
        if self.flash_fft and False:
            U_plus, U_minus = flash_convolve(
                x, self.phi, self.flash_fft, self.use_approx
            )
        else:
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




def get_hankel(seq_len: int) -> torch.Tensor:
    # Create a tensor with values from 1 to seq_len (inclusive)
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64).to('cuda')
    # Compute the outer sum
    i_plus_j = entries[:, None] + entries[None, :]
    # Calculate Z using element-wise operations
    Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int, 
    K: int, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Compute the Hankel matrix using PyTorch
    Z = get_hankel(seq_len)
    # Compute eigenvalues and eigenvectors for symmetric matrices
    sigma, phi = torch.linalg.eigh(Z)
    # Select the largest K eigenvalues and corresponding eigenvectors
    sigma, phi = sigma[-K:], phi[:, -K:]
    # Scale the eigenvectors with the eigenvalues raised to 0.25 (broadcasting applies)
    phi = phi * (sigma ** 0.25)
    # Return the tensor on the desired device and with the desired data type
    return phi.to(device=device, dtype=dtype)


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

    return U_plus.to(u.dtype), U_minus.to(u.dtype)

def flash_convolve(
    u: torch.Tensor, v: torch.Tensor, flash_fft, use_approx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if use_approx:
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

    U_conv = flash_fft(u_conv, v_padded)
    # Trim the output back to the original sequence length
    U_conv = U_conv[..., :seq_len]

    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

    if use_approx:
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus, U_minus

phi = None
@torch.no_grad
def build_phi(seqlen: int = 16 * 1024, K: int = K, device: str = 'cuda'):
    global phi
    phi = get_spectral_filters(seqlen, K=K, device=device, dtype=torch.float32)
    return phi

if __name__ == '__main__':
    seq_len = 256
    n_embd = 64
    device = 'cuda'
    torch_dtype = torch.float32
    n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
    phi = get_spectral_filters(seq_len, K, device, torch_dtype)

    layer = STU(
        n_embd=n_embd,
        idx=0,
        torch_dtype=torch_dtype,
        phi=phi,
        n=n,
        gating=True,
    ).to(device)

    x = torch.randn(2, seq_len, n_embd).to(device)

    out, mem = layer(x, None)
    print(out.shape)
    print(mem.shape if mem is not None else 0)