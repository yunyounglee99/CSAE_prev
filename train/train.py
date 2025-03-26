import torch
import torch.nn.functional as F
from torch import optim
from models.backbone import ViTBackbone
from models.sae import SparseAutoencoder
from models.classifier import ClassifierHead
from models.csae import CSAE
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

  model = CSAE(vit_model=vit_model, latent_dim=latent_dim, topk=topk, num_classes=100, num_parallel_layers=5)
  model.to(device)
  model.backbone.eval()

  optimizer = optim.Adam(list(sae.parameters()) + list(classifier.parameters()), lr = lr)
  train_loaders, test_loaders = dataloaders(batch_size=batch_size, num_tasks=num_tasks)

  for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    print(f"\n=================Task {task_id}==================================")
    model.sae.train()
    model.classifier.train()
    loss = 0.0

    for epoch in range(epochs):
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, latent_list = model(images)
        loss_ce = F.cross_entropy(logits, labels)
        loss_l1 = torch.mean(torch.abs(latent_list[-1])
                             )

