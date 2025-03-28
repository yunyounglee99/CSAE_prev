import torch
import torch.nn as nn

class SaeWrapper(nn.Module):
  def __init__(self, vit_layer, sae_module):
    super().__init__()
    self.layer = vit_layer
    self.sae = sae_module

  def forward(self, x, prev_weight = None, task_id = 0):
    x = self.layer(x)[0]
    mod, latent, current_weight = self.sae(x, prev_weight, task_id)

    return x+0.1 * mod, latent, current_weight