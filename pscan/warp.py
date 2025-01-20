from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from torch import Tensor

cuda_source = (Path(__file__).parent / 'csrc' / 'warp.cuh').read_text()

cpp_source = """
at::Tensor warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse);
"""

module = load_inline(
    name='warpy',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['warpscan_forward'],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--ptxas-options=-v",
        "-lineinfo",
        "--fmad", "false",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
)

@torch.library.custom_op("mylib::warpscan_forward", mutates_args=(), device_types="cuda")
def warpscan_forward(gates: Tensor, tokens: Tensor, reverse: bool = False) -> Tensor:
    output = torch.zeros_like(tokens)
    module.warpscan_forward(gates, tokens, output, reverse)
    return output

def warpscan_forwardfake(gates: Tensor, tokens: Tensor, reverse: bool = False):
    # torch._check(gates.device == tokens.device)
    # torch._check(gates.dtype == tokens.dtype)
    return torch.empty_like(gates)

warpscan_forward.register_fake(warpscan_forwardfake)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, tokens):
        B, C, T = gates.shape
        assert tokens.shape == (B, C, T)
        assert gates.is_contiguous()
        assert tokens.is_contiguous()

        states = warpscan_forward(gates, tokens)
        ctx.save_for_backward(states, gates)
        return states

    # backward scan is a padded reverse scan
    # See https://arxiv.org/abs/1709.04057 Section 2.2
    @staticmethod
    def backward(ctx, grad_output):
        states, gates = ctx.saved_tensors
        B, C, T = gates.shape

        grad_output = grad_output.contiguous()
        assert states.is_contiguous()
        assert gates.is_contiguous()

        padded_shifted_gates = torch.cat([gates, torch.ones_like(gates[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
        d_states = warpscan_forward(padded_shifted_gates, grad_output, reverse=True)

        padded_outputs = torch.cat([torch.zeros_like(states[:, :, :1]), states], dim=-1)[:, :, :-1]
        d_gates = padded_outputs * d_states

        d_tokens = d_states
        return d_gates, d_tokens


def scan(gates, tokens):
    """Solve a first-order recurrence relation:

    .. math::
        x_t = a_t x_{t-1} + b_t

    where :math:`a_t` ("gates") and :math:`b_t` ("tokens") are sequences of vectors.

    Arguments:
        gates (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.
        tokens (torch.Tensor): shape (B, C, T), must be contiguous. T must be a power of 2.

    Returns:
        (torch.Tensor): shape (B, C, T)
    """
    return Scan.apply(gates, tokens)


if __name__ == "__main__":
    # Example to test if it works with torch.compile
    @torch.compile()
    def test_scan(gates, tokens):
        return scan(gates, tokens)

    # Test the functionality
    gates = torch.randn(2, 3, 64).cuda()
    tokens = torch.randn(2, 3, 64).cuda()

    output = test_scan(gates, tokens)
    print(output)