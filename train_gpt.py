import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6 8.9'
import torch
from pscan.warp import scan

import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from pscan.warp import scan
import math
import random
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(1234)
torch._inductor.config.coordinate_descent_tuning = True # turn this off for a faster compile time (but slightly slower run)
use_lsgm = True

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

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


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

class LSGM(nn.Module):
    def __init__(self,
        dim: int,
        expansion_factor: int = 1.5,
        kernel_size: int = 4,
        num_slots: int = 16,
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
            self.write_query = Linear(slot_dim, slot_dim, bias=False)
            self.write_key = Linear(hidden, slot_dim, bias=False)
            self.write_value = Linear(hidden, slot_dim, bias=False)
            self.write_gate = Linear(hidden, 1, bias=False)
            self.read_query = Linear(hidden, slot_dim, bias=False)
            self.read_key = Linear(slot_dim, slot_dim, bias=False)
            self.read_value = Linear(slot_dim, hidden, bias=False)
        self.segment_length = block_size // num_slots
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        with torch.no_grad():
            self.input.weight.normal_(std=dim ** -0.5)
            self.gates.weight.normal_(std=hidden ** -0.5)
            self.output.weight.normal_(std=hidden ** -0.5)
            if mem_enhance:
                nn.init.normal_(self.write_query.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.write_key.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.write_value.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.write_gate.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.read_query.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.read_key.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.read_value.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.memory_slots, std=0.02)

    def write_memory(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots

        # Reshape x into segments
        x_segments = x.view(B, self.num_slots, segment_length, -1)

        # Query, Key, and Value
        q = self.write_query(self.memory_slots.to(x.dtype))  # [num_slots, slot_dim]
        k = self.write_key(x_segments)  # [B, num_slots, segment_length, slot_dim]
        v = self.write_value(x_segments)  # [B, num_slots, segment_length, slot_dim]

        # Expand q to match B dimension
        q = q.unsqueeze(0).expand(B, -1, -1)  # [B, num_slots, slot_dim]

        # Each slot attends only to its corresponding segment
        # Reshape k and v to match the B dimension
        k = k.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, slot_dim]
        v = v.view(B, self.num_slots, segment_length, -1)  # [B, num_slots, segment_length, slot_dim]

        # Compute attention for each slot and segment
        qk = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / math.sqrt(self.slot_dim)  # [B, num_slots, 1, segment_length]
        attn = F.softmax(qk, dim=-1)  # [B, num_slots, 1, segment_length]

        # Memory computation (without loop)
        memory = torch.matmul(attn, v)  # [B, num_slots, 1, slot_dim]
        memory = memory.squeeze(2)  # [B, num_slots, slot_dim]

        return memory

    def read_memory(self, x: Tensor, memory: Tensor) -> Tensor:
        B, T, _ = x.shape
        segment_length = T // self.num_slots
        # Query, Key, and Value
        q = self.read_query(x)      # [B, T, slot_dim]
        k = self.read_key(memory)   # [B, num_slots, slot_dim]
        v = self.read_value(memory) # [B, num_slots, dim]
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
        # Apply gate
        output = F.sigmoid(self.write_gate(x)) * output
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
            x = self.read_memory(F.gelu(x) * h, mem)
            return self.output(x)

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

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()

        self.attn = CausalSelfAttention(dim, num_heads) if use_lsgm is False else LSGM(dim)
        self.mlp = MLP(dim) #if use_lsgm is False else GatedMLP(dim)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class Model(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.lm_head = Linear(model_dim, next_multiple_of_n(vocab_size, n=128))
        nparams = self.num_params() / 1e6
        print0("Number of parameters: %.3fM" % (nparams,))
        print("Number of parameters: %.3fM" % (nparams,))

    def num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.embed.weight.numel()
        return n_params

    def forward(self, input_seq: Tensor, target_seq: Tensor):
        x = self.embed(input_seq)[None]
        for block in self.blocks:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss

def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn"t helpful.
        pos += batch_size
        yield inputs, targets

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization
    # batch_size = 8*64*1024 # batch size in tokens
    batch_size = 64 * 1024 # batch size in tokens
    num_iterations = 1010 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    # implementation
    seq_len = 16 * 1024
    save_checkpoint = False
args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
             if console:
                 print(s)
             print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

# load data
train_loader = distributed_data_generator(args.train_files, args.batch_size, rank, world_size)

model = Model(vocab_size=50257, num_layers=8, num_heads=2, model_dim=128).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)


optimizer = torch.optim.Adam(model.parameters(), fused=True)

def get_lr(it: int):
    t = 1 - it / args.num_iterations
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
    return w * 1.0 + (1 - w) * 0.1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)


# model: nn.Module = torch.compile(model)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float("nan") if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val


    window_size = next_multiple_of_n(1728 * step / train_steps, n=128)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_bs = world_size * args.seq_len
        assert args.val_tokens % val_bs == 0
        val_steps = 50 # args.val_tokens // val_bs
        val_loader = distributed_data_generator(args.val_files, val_bs, rank, world_size)
        val_loss = 0

        with torch.no_grad():
            for s in range(val_steps):
                print(s, '/', val_steps, end='\r')
                x, y = next(val_loader)
                val_loss += model(x, y)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        del val_loader

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")

        break

    # --------------- TRAINING SECTION BEGIN -----------------
    inputs, targets = next(train_loader)
    for input_seq, target_seq in zip(inputs.split(args.seq_len), targets.split(args.seq_len)):
        train_loss = model(input_seq, target_seq)
        train_loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    optimizer.step()
    scheduler.step()
    model.zero_grad(set_to_none=True)
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms", console=True)

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
)
dist.destroy_process_group()
if master_process:
    print(run_id)