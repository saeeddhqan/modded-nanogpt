import stu
from util import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(1234)

device = 'cpu'
dtype = torch.float32


def generate_synthetic_data(seqlen, dim, nsamples: int = 15):
    x_data = torch.randn(nsamples, 1, seqlen, dim)
    y_data = torch.randint(128, (nsamples, 1, seqlen))  # dummy targets
    return list(zip(x_data, y_data))

phi = None
def train_model(dim, seqlen, nlayers, K, epochs=1, lr=1e-3):
    model = stu.STUModel(
        dim=dim,
        seqlen=seqlen,
        nlayers=nlayers,
        K=K,
        vocab_size=128,
        use_gating=True,
        device=device,
        dtype=dtype,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    data = generate_synthetic_data(seqlen, dim)
    losses = []
    grad_norms = []
    output_means = []
    logits_means = []
    layers_means = []
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in data:
            optimizer.zero_grad(set_to_none=True)
            loss, logits, mean_layers = model(x_batch, y_batch)
            loss.backward()
            total_norm = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    # print('grad:\t', param_norm, ',', name)
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            losses.append(round(loss.item(), 5))
            grad_norms.append(round(total_norm, 5))
            logits_means.append(round(logits.mean().item(), 4))
            layers_means.append(mean_layers)
            optimizer.step()
            # print(output_means[-1], end='\r')
            # print('----')
        # exit()
    layers_means = torch.tensor(layers_means).mean(dim=0).cpu().tolist()
    # if hasattr(model.attn[0], 'cross_attn'):
        # print(model.attn[0].cross_attn.write_matter)
    # exit()
    return losses, grad_norms, logits_means, layers_means

dims = [32]
seqlens = [32]
# nlayers_list = [1, 2, 3, 4, 5, 6]
# nlayers_list = [2,3,4, 8, 32, 64, 128]
nlayers_list = [4]
Ks = [8]
results = {}  # to store results for each combination

for dim_ in dims:
    for seqlen_ in seqlens:
        for nlayers_ in nlayers_list:
            for K_ in Ks:
                print(f"Training model with dim={dim_}, seqlen={seqlen_}, nlayers={nlayers_}, K={K_}...")
                losses, grad_norms, logits_means, layers_means = train_model(dim_, seqlen_, nlayers_, K_, epochs=2)
                
                key = f"dim={dim_}, seqlen={seqlen_}, nlayers={nlayers_}, K={K_}"
                results[key] = {
                    'losses': losses,
                    'grad_norms': grad_norms,
                    'logits_means': logits_means,
                    'layers_means': layers_means,
                }

with open("test_stu.json", 'w') as fp:
    fp.write(json.dumps(results, indent=4))

for i, (key, metrics) in enumerate(results.items()):
    fig, (ax_loss, ax_grad, ax_logit, ax_layer) = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    
    # Plot losses
    ax_loss.plot(metrics['losses'], label='loss')
    ax_loss.set_title(f"Loss: {key}")
    ax_loss.set_xlabel('Steps')
    ax_loss.set_ylabel('Loss')
    
    # Plot gradient norm
    ax_grad.plot(metrics['grad_norms'], label='grad_norm', color='orange')
    ax_grad.set_title(f"Grad norm: {key}")
    ax_grad.set_xlabel('Steps')
    ax_grad.set_ylabel('Grad Norm')
    
    # Plot logits mean
    ax_logit.plot(metrics['logits_means'], label='logits_mean', color='black')
    ax_logit.set_title(f"Logits mean: {key}")
    ax_logit.set_xlabel('Steps')
    ax_logit.set_ylabel('Mean of logits')

    # Plot logits mean
    ax_layer.plot(metrics['layers_means'], label='layers_mean', color='black')
    ax_layer.set_title(f"Layers mean: {key}")
    ax_layer.set_xlabel('Steps')
    ax_layer.set_ylabel('Mean of layers')

    plt.tight_layout()
    plt.show()
