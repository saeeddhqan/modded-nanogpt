import os
from util import *
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6 8.9'
import torch
from LSGM import LSGM
from attention import CausalSelfAttention
from stu import STU, phi
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
method = 'attn'



class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        seqlen = 16 * 1024
        if method == 'attn':
            self.attn = CausalSelfAttention(dim, num_heads)
        elif method == 'lsgm':
            self.attn = LSGM(dim)
        elif 'stu' in method:
            n = nearest_power_of_two(seqlen * 2 - 1, round_up=True)
            self.attn = STU(
                n_embd=dim,
                num_eigh=24,
                torch_dtype=torch.float32,
                phi=phi,
                n=n,
                gating=True if 'gating' in method else False,
            ).to(device)
        else:
            raise Exception("method not found")
        self.mlp = MLP(dim) if method != 'lsgm' else GatedMLP(dim)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


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
