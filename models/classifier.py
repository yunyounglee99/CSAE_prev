import torch.nn as nn

class ClassifierHead(nn.Module):
  def __init__(self, input_dim = 768, num_classes = 100):
    super().__init__()
    self.classifier = nn.Linear(input_dim, num_classes)

  def forward(self, x):
    return self.classifier(x)