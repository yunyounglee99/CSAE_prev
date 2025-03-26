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

  optimizer = optim.Adam(list(model.sae.parameters()) + list(model.classifier.parameters()), lr = lr)
  train_loaders, test_loaders = dataloaders(batch_size=batch_size, num_tasks=num_tasks)

  for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    print(f"\n=================Task {task_id}==================================")
    model.sae.train()
    model.classifier.train()
    total_loss = 0.0

    for epoch in range(epochs):
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, latent_list = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        l1_loss = torch.mean(torch.abs(latent_list[-1]))
        loss = ce_loss + l1_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      print(f'Task {task_id} Epoch {epoch+1}/{epochs} loss : {total_loss/len(train_loader):.4f}')
      total_loss = 0.0

    checkpoint = {"csae" : model.state_dict()}
    torch.save(checkpoint, f"checkpoint_task{task_id}.pth")

    model.sae.eval()
    model.classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct /total
    print(f"Task {task_id} acc : {acc:.2f}%")

if __name__ == "__main__":
  train()
