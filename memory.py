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
        num_slots: int = 8,
        num_heads: int = 1,
        block_size: int = 65536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.idx, self.dim, self.num_slots = idx, dim, num_slots
        self.segment_length = block_size // num_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        assert num_slots <= block_size, "invalid num slots"

        # Create read projections
        self.read_q = Linear(dim, dim, bias=False)
        self.read_kv = Linear(dim, dim * 2, bias=False)
        # Only the first instance gets write projections
        if idx == 0:
            self.memory_slots = nn.Parameter(torch.randn(num_slots, dim))
            self.write_q, self.write_k, self.write_v = tuple(
                Linear(dim, dim, bias=False) for _ in range(3)
            )
            self.write_matter = nn.Parameter(torch.ones(dim) * 0.01)

        # Initialize parameters
        with torch.no_grad():
            for layer in (self.read_q, self.read_kv):
                nn.init.normal_(layer.weight, std=0.02)
            if idx == 0:
                nn.init.normal_(self.memory_slots, std=0.02)
                for layer in (self.write_q, self.write_k, self.write_v):
                    nn.init.normal_(layer.weight, std=0.02)

    def write_memory(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots
        
        x_segments = x.view(B, self.num_slots, segment_length, -1)

        # Project to multi-head key and value
        k = self.write_k(x_segments)  # [B, num_slots, segment_length, dim]
        v = self.write_v(x_segments)  # [B, num_slots, segment_length, dim]
        
        # Reshape k and v to separate heads
        k = k.view(B, self.num_slots, segment_length, self.num_heads, self.head_dim)
        v = v.view(B, self.num_slots, segment_length, self.num_heads, self.head_dim)
        
        # Project memory slots to queries
        q = self.write_q(self.memory_slots.to(x.dtype))  # [num_slots, dim]
        q = q.view(self.num_slots, self.num_heads, self.head_dim)  # [num_slots, num_heads, head_dim]
        q = q[None,...].expand(B, -1, -1, -1)  # [B, num_slots, num_heads, head_dim]

        # Rearrange dimensions for attention
        k = k.permute(0, 3, 1, 2, 4)  # [B, num_heads, num_slots, segment_length, head_dim]
        v = v.permute(0, 3, 1, 2, 4)  # [B, num_heads, num_slots, segment_length, head_dim]
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, num_slots, head_dim]

        # Add necessary dimensions for attention
        q = q.unsqueeze(3)  # [B, num_heads, num_slots, 1, head_dim]

        memory = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, num_heads, num_slots, 1, head_dim]

        # Reshape and combine heads
        memory = memory.squeeze(3)  # [B, num_heads, num_slots, head_dim]
        memory = memory.permute(0, 2, 1, 3)  # [B, num_slots, num_heads, head_dim]
        memory = memory.reshape(B, self.num_slots, -1)  # [B, num_slots, dim]

        return norm(memory * self.write_matter)

    def read_memory(self, x: Tensor, memory: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots
        # Query, Key, and Value
        q = self.read_q(x)      # [B, T, dim]
        k, v = self.read_kv(memory).chunk(2, dim=-1) # [B, num_slots, dim]
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
        output = torch.matmul(attn, v) # [B, T, dim]
        return output

    def forward(self, x: Tensor, memory: Tensor | None) -> Tensor:
        if self.idx == 0:
            memory = self.write_memory(x)
        x = self.read_memory(x, memory)
        return F.silu(x), memory

if __name__ == "__main__":
    # check the output of both write methods
    model = memory(64, 0, block_size=128)
    dummy = torch.randn(1, 128, 64)
    out1 = model.write_memory(dummy)
