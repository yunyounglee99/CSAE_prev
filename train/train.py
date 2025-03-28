# train/train.py
import sys
import os

# 현재 파일(train.py)의 상위 디렉터리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch import optim
from models.backbone import ViTBackbone
from models.sae import SparseAutoencoder
from models.classifier import ClassifierHead
from models.csae import CSAE
from dload import dataloaders
from tqdm import tqdm

def train_csae(
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
  print(device)

  model = CSAE(vit_model=vit_model, latent_dim=latent_dim, topk=topk, num_classes=100, num_parallel_layers=5)
  model.to(device)
  model.backbone.eval()

  optimizer = optim.Adam(list(model.sae.parameters()) + list(model.classifier.parameters()), lr = lr)
  train_loaders, test_loaders = dataloaders(batch_size=batch_size, num_tasks=num_tasks)

  prev_weight = None
  
  for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    print(f"\n=================Task {task_id}==================================")
    model.sae.train()
    model.classifier.train()
    total_loss = 0.0

    for epoch in tqdm(range(epochs)):
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, latent_list, current_weight = model(images, prev_weight = prev_weight)
        ce_loss = F.cross_entropy(logits, labels)
        l1_loss = torch.mean(torch.abs(latent_list[-1]))
        loss = ce_loss + l1_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      print(f'Task {task_id} Epoch {epoch+1}/{epochs} loss : {total_loss/len(train_loader):.4f}')
      total_loss = 0.0

    prev_weight = current_weight

    checkpoint = {"csae" : model.state_dict()}
    torch.save(checkpoint, f"checkpoint_task{task_id}.pth")
    print(f"task {task_id} is saved : checkpoint_task{task_id}.pth")

    model.sae.eval()
    model.classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct /total
    print(f"Task {task_id} acc : {acc:.2f}%")

if __name__ == "__main__":
  train_csae()
