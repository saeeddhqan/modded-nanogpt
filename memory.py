from util import *
import math
import torch
nn = torch.nn
F = nn.functional

class memory(nn.Module):
    # cross attn
    def __init__(self,
        dim: int,
        idx: int,
        num_slots: int = 32,
        block_size: int = 65536,
    ):
        super().__init__()
        self.idx, self.dim, self.num_slots = idx, dim, num_slots
        self.segment_length = block_size // num_slots

        self.gate = Linear(dim, dim)
        
        # Create read projections
        self.read_q, self.read_k, self.read_v = tuple(
            Linear(dim, dim, bias=False) for _ in range(3)
        )
        self.output = Linear(dim, dim, bias=False)
        # Only the first instance gets write projections
        if idx == 0:
            self.memory_slots = nn.Parameter(torch.randn(num_slots, dim))
            self.write_q, self.write_k, self.write_v = tuple(
                Linear(dim, dim, bias=False) for _ in range(3)
            )

        # Initialize parameters
        with torch.no_grad():
            self.gate.weight.normal_(std=dim ** -0.5)
            self.output.weight.normal_(std=dim ** -0.5)
            for layer in (self.read_q, self.read_k, self.read_v):
                nn.init.normal_(layer.weight, std=0.02)
            if idx == 0:
                nn.init.normal_(self.memory_slots, std=0.02)
                for layer in (self.write_q, self.write_k, self.write_v):
                    nn.init.normal_(layer.weight, std=0.02)

    def write_memory(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots

        # Reshape x into segments
        x_segments = x.view(B, self.num_slots, segment_length, -1)

        # Query, Key, and Value
        q = self.write_q(self.memory_slots.to(x.dtype))  # [num_slots, dim]
        k = self.write_k(x_segments)  # [B, num_slots, segment_length, dim]
        v = self.write_v(x_segments)  # [B, num_slots, segment_length, dim]

        # Expand q to match B dimension
        q = q.unsqueeze(0).expand(B, -1, -1)  # [B, num_slots, dim]

        # Each slot attends only to its corresponding segment
        # Reshape k and v to match the B dimension
        k = k.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, dim]
        v = v.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, dim]

        # Compute attention for each slot and segment
        qk = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / math.sqrt(self.dim)  # [B, num_slots, 1, segment_length]
        attn = F.softmax(qk, dim=-1)  # [B, num_slots, 1, segment_length]

        memory = torch.matmul(attn, v)  # [B, num_slots, 1, dim]
        memory = memory.squeeze(2)  # [B, num_slots, dim]

        return memory

    def read_memory(self, x: Tensor, memory: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots
        # Query, Key, and Value
        q = self.read_q(x)      # [B, T, dim]
        k = self.read_k(memory)   # [B, num_slots, dim]
        v = self.read_v(memory) # [B, num_slots, dim]
        # Compute attention scores
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)  # [B, T, num_slots]
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

    def forward(self, x: Tensor, memory: Tensor | None) -> Tensor:
        h = self.gate(x)
        if self.idx == 0:
            memory = self.write_memory(x)
        x = x + F.sigmoid(h) * self.read_memory(x, memory)
        return self.output(x), memory
