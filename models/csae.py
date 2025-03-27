import torch
import torch.nn as nn
from models.backbone import ViTBackbone
from models.sae import SparseAutoencoder
from models.sae_wrapper import SaeWrapper
from models.classifier import ClassifierHead

class CSAE(nn.Module):
  def __init__(
      self,
      vit_model = "google/vit-base-patch16-224-in21k",
      latent_dim = 256,
      topk = 50,
      num_classes = 100,
      num_parallel_layers = 4):
    super().__init__()
    self.backbone = ViTBackbone(model_name=vit_model)
    self.hidden_dim = self.backbone.hidden_dim 
    self.sae = SparseAutoencoder(input_dim=self.hidden_dim, latent_dim=latent_dim, topk=topk)
    self.num_parallel_layers = num_parallel_layers  
    self.classifier = ClassifierHead(input_dim=self.hidden_dim, num_classes=num_classes)

  def forward(self, images, prev_weight = None):
    x = self.backbone(images)
    latent_list = []

    current_weight = prev_weight

    for i in range(self.num_parallel_layers):
      layer = self.backbone.vit.encoder.layer[i]
      wrapper = SaeWrapper(layer, self.sae)
      x, latent, current_weight = wrapper(x, prev_weight = current_weight)
      latent_list.append(latent)

    for i in range(self.num_parallel_layers, len(self.backbone.vit.encoder.layer)):
      x = self.backbone.vit.encoder.layer[i](x)[0]

    x = self.backbone.vit.layernorm(x)
    x = self.backbone.vit.pooler(x)
    logits = self.classifier(x)
    return logits, latent_list, current_weight