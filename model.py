
import torch
from torch import nn
from util import *
from attention import CausalSelfAttention
import stu

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, idx: int, method: str, seqlen: int, is_causal: bool, **further_params):
        super().__init__()
        self.method = method
        if method == 'attn':
            self.attn = CausalSelfAttention(dim, num_heads, idx, seqlen, is_causal, further_params["num_slots"], further_params["use_gating"])
        elif 'stu' in method:
            device = further_params.pop('device')
            dtype = further_params.pop('dtype')
            self.attn = stu.STU(
                n_embd=dim,
                idx=idx,
                torch_dtype=dtype,
                is_causal=is_causal,
                n=seqlen,
                phi=stu.get_spectral_filters(seqlen, further_params['K'], device=device, dtype=dtype),
                **further_params
            ).to(device)
        else:
            raise Exception("method not found")
        self.mlp = MLP(dim) if method != 'lsgm' else GatedMLP(dim)

    def forward(self, x, mem):
        y, mem = self.attn(norm(x), mem)
        x = x + y
        x = x + self.mlp(norm(x))
        return x, mem


class Model(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, method: str, seqlen: int, is_causal: bool, **further_params):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, idx, method, seqlen, is_causal, **further_params) for idx in range(num_layers)])
        self.lm_head = Linear(model_dim, vocab_size)
        self.embed.weight = self.lm_head.weight
        nparams = self.num_params() / 1e6
        self.apply(self.norm_weights)
        # print0("Number of parameters: %.3fM" % (nparams,))
        print("Number of parameters: %.3fM" % (nparams,))

    def num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.embed.weight.numel()
        return n_params

    def norm_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_seq: Tensor, target_seq: Tensor = None):
        x = self.embed(input_seq)
        mem = None
        for block in self.blocks:
            x, mem = block(x, mem)
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5)) # B, T, vocab_size
        loss = None
        if isinstance(target_seq, Tensor):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1)) # target_seq shape (B, T)
        return loss, logits
