import torch
import torch.nn as nn

class SaeWrapper(nn.Module):
  def __init__(self, vit_layer, sae_module):
    super().__init__()
    self.layer = vit_layer
    self.sae = sae_module

  def forward(self, x):
    x = self.layer(x)[0]
    mod, latent = self.sae

    return x+mod, latent