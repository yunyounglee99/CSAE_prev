import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
  def __init__(self, input_dim = 768, latent_dim = 256, topk = 50):
    super().__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.topk = topk
    self.encoder = nn.Linear(input_dim, latent_dim)
    self.decoder = nn.Linear(latent_dim, input_dim)
    self.alpha = nn.Parameter(torch.tensor(1.0))

  def forward(self, x, prev_weight = None):
    B, L, D = x.size()
    x_flat = x.view(B*L, D)

    latent = self.encoder(x_flat)

    if self.topk is not None and self.topk < self.latent_dim:
      topk_vals, topk_idx = torch.topk(latent.abs(), k=self.topk, dim = 1)
      mask = torch.zeros_like(latent)
      mask.scatter_(1, topk_idx, 1)
      gated_latent = latent * mask
    else:
      gated_latent = latent
    if prev_weight is not None:
      prev_weight.requires_grad = False
      delta = self.decoder.weight - prev_weight
      W = prev_weight + self.alpha * delta
    else:
      W = self.decoder.weight
    
    mod_flat = gated_latent @ W.T
    mod = mod_flat.view(B, L, D)

    return mod, latent, W