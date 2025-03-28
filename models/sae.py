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
    self.s = nn.Parameter(torch.tensor(0.0))

  def get_alpha(self, task_id):
    if task_id == 0:
      return torch.tensor(1.0, device = self.s.device)
    else:
      lower_bound = 0.0
      upper_bound = 1.0 / (task_id + 1)
      alpha = lower_bound + upper_bound * torch.sigmoid(self.s)
      return alpha
    
  def forward(self, x, prev_weight = None, task_id = 0):
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
      prev_weight = prev_weight.detach()
      delta = self.decoder.weight - prev_weight
      alpha = self.get_alpha(task_id)
      W = prev_weight + alpha * delta
    else:
      W = self.decoder.weight
    
    mod_flat = gated_latent @ W.T
    mod = mod_flat.view(B, L, D)

    return mod, latent, W