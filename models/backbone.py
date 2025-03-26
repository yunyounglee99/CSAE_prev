import torch
import torch.nn as nn
from transformers import ViTModel

class ViTBackbone(nn.Module):
  def __init__(self, model_name = "google/vit-base-patch16-224-in21k"):
    super().__init__()
    self.vit = ViTModel.from_pretrained(model_name)
    for param in self.vit.parameters():
      param.requires_grad = False
    self.hidden_dim = self.vit.config.hidden_size

  def forward(self, images):
    embeddings = self.vit.embeddings(images)
    return embeddings