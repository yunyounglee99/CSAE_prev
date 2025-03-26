import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
  def __init__(self, input_dim = 768, latent_dim = 256, toopk = 50):
    super().__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.topk = toopk
    self.encoder = nn.Linear(input_dim, latent_dim)
    self.decoder = nn.Linear(latent_dim, input_dim)

  def forward(self, x):
    B, L, D = x.size()
    x_flat = x.view(B*L, D)
    latent = self.encoder(x_flat)
    latent = torch.relu(latent)

    if self.topk is not None and self.topk < self.latent_dim:
      topk_vals, topk_idx = torch.topk(latent.abs(), k=self.topk, dim = 1)
      mask = torch.zeros_like(latent)
      mask.scatter_(1, topk_idx, 1)
      gated_latent = latent *mask
    else:
      gated_latent = latent

    mod_flat = self.decoder(gated_latent)
    mod = mod_flat.view(B, L, D)

    return mod, latent