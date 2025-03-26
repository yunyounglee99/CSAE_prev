import torch
import torch.nn.functional as F
from torch import optim
from models.backbone import ViTBackbone
from models.sae import SparseAutoencoder
from models.classifier import ClassifierHead
from dataloader import dataloaders

def train(
    num_tasks = 10,
    batch_size = 128,
    epochs = 20,
    lr = 5e-5,
    latent_dim = 256,
    topk = 50,
    l1_coeff = 1e-4,
    vit_model = "google/vit-base-patch16-224-in21k"
):
  if torch.cuda.is_available():
    device_name = "cuda"
  elif torch.mps.is_available():
    device_name = "mps"
  else:
    divice_name = "cpu"
  device = torch.device(device_name)

  backbone = ViTBackbone(model_name = vit_model).to(device)
  sae = SparseAutoencoder(input_dim=backbone.hidden_dim, latent_dim=latent_dim, topk=topk).to(device)
  classifier = ClassifierHead(input_dim = backbone.hidden_dim, num_classes=100).to(device)

  backbone.eval()

  optimizer = optim.Adam(list(sae.parameters()) + list(classifier.parameters()), lr = lr)
  task_loaders = dataloaders(batch_size=batch_size, num_tasks=num_tasks)
