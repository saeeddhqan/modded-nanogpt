from util import *
import math
import torch
from pscan.warp import scan
nn = torch.nn
F = nn.functional

class LSGM(nn.Module):
	# Linear sequential gated memory
    def __init__(self,
        dim: int,
        expansion_factor: int = 1.5,
        kernel_size: int = 4,
        num_slots: int = 32,
        slot_dim: int = 384,
        block_size: int = 65536,
        lsg: bool = False,
        mem_enhance: bool = True,
    ):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = Linear(dim, 2 * hidden, bias=False)
        self.conv = Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size - 1)
        self.gates = Linear(hidden, 2 * hidden, bias=True)
        self.lsg = lsg
        if lsg:
            bases = []
            for x in range(num_slots):
                bases.append(torch.linspace(-4.323, -9, hidden))
            self.forget_base = nn.Parameter(torch.cat(bases).view(1, num_slots, 1, -1))
        else:
            self.forget_base = nn.Parameter(torch.linspace(-4.323, -9, hidden))
        self.output = Linear(hidden, dim, bias=False)
        self.mem_enhance = mem_enhance
        if mem_enhance:
            self.memory_slots = nn.Parameter(torch.randn(num_slots, slot_dim))
            self.write_q = Linear(slot_dim, slot_dim, bias=False)
            self.write_kv = Linear(hidden, slot_dim, bias=False)
            self.read_q = Linear(hidden, slot_dim, bias=False)
            self.read_kv = Linear(slot_dim, slot_dim, bias=False)
            self.read_kv = Linear(slot_dim, hidden, bias=False)
        self.segment_length = block_size // num_slots
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        with torch.no_grad():
            self.input.weight.normal_(std=dim ** -0.5)
            self.gates.weight.normal_(std=hidden ** -0.5)
            self.output.weight.normal_(std=hidden ** -0.5)
            if mem_enhance:
                nn.init.normal_(self.write_q.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.write_kv.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.read_q.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.read_kv.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.memory_slots, std=0.02)

    def write_memory(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots

        # Reshape x into segments
        x_segments = x.view(B, self.num_slots, segment_length, -1)

        # Query, Key, and Value
        q = self.write_q(self.memory_slots.to(x.dtype))  # [num_slots, slot_dim]
        k = self.write_kv(x_segments)  # [B, num_slots, segment_length, slot_dim]
        v = self.write_kv(x_segments)  # [B, num_slots, segment_length, slot_dim]

        # Expand q to match B dimension
        q = q.unsqueeze(0).expand(B, -1, -1)  # [B, num_slots, slot_dim]

        # Each slot attends only to its corresponding segment
        # Reshape k and v to match the B dimension
        k = k.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, slot_dim]
        v = v.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, slot_dim]

        # Compute attention for each slot and segment
        qk = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / math.sqrt(self.slot_dim)  # [B, num_slots, 1, segment_length]
        attn = F.softmax(qk, dim=-1)  # [B, num_slots, 1, segment_length]

        memory = torch.matmul(attn, v)  # [B, num_slots, 1, slot_dim]
        memory = memory.squeeze(2)  # [B, num_slots, slot_dim]

        return memory

    def read_memory(self, x: Tensor, memory: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots
        # Query, Key, and Value
        q = self.read_q(x)      # [B, T, slot_dim]
        k = self.read_kv(memory)   # [B, num_slots, slot_dim]
        v = self.read_kv(memory) # [B, num_slots, dim]
        # Compute attention scores
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.slot_dim)  # [B, T, num_slots]
        # Create causal mask for slots
        # Each token i can only attend to slots j where j <= i//segment_length
        mask = torch.arange(T, device=x.device).unsqueeze(1) // segment_length  # [T, 1] -> Which segment the token belongs to
        mask = mask <= torch.arange(self.num_slots, device=x.device)  # [T, num_slots] -> Causal mask for each slot
        # Apply causal mask: Replace where mask is 0 with -inf to prevent attention
        qk = qk.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))  # [B, T, num_slots]
        # Attention and output
        attn = F.softmax(qk, dim=-1)  # [B, T, num_slots]
        output = torch.matmul(attn, v)  # [B, T, dim]

        return output

    def forward(self, x: Tensor) -> Tensor:
        if self.lsg:
            return self.lsg_forward(x)
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :x.size(1)].mT
        forget, inp = self.gates(x).chunk(2, dim=-1)
        alpha = (-8 * F.softplus(self.forget_base.to(x.dtype)) * forget.sigmoid()).exp()
        x = (1 - alpha ** 2 + 1e-6).sqrt() * inp.sigmoid() * x
        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT

        if self.mem_enhance:  # this is a divergence from hawk
            x = gate * gate.sigmoid() * x
            mem = self.write_memory(x)
            x = x + self.read_memory(x, mem)
            return self.output(F.gelu(x) * h)

        return self.output(F.gelu(gate) * h)

    def lsg_forward(self, x: Tensor) -> Tensor:
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :x.size(1)].mT
        forget, inp = self.gates(x).chunk(2, dim=-1)
        B, T, C = forget.size()
        forget = forget.view(B, self.num_slots, self.segment_length, C)

        alpha = (-8 * F.softplus(self.forget_base.to(x.dtype)) * forget.sigmoid()).exp()
        alpha = alpha.view(B, T, C).contiguous()
        beta = (1 - alpha ** 2 + 1e-6).sqrt()
        x = beta * inp.sigmoid() * x
        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        if self.mem_enhance:  # this is a divergence from hawk
            x = gate * gate.sigmoid() * x
            mem = self.write_memory(x)
            x = x + self.read_memory(x, mem)
            return self.output(F.gelu(x) * h)
        x = self.output(F.gelu(gate) * h)
        return x
