from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def select_class_subset(dataset, class_list):
  indices = [i for i, target in enumerate(dataset.targets) if target in class_list]
  return Subset(dataset, indices)

def create_task_dataloaders(train_dataset, test_dataset, task_splits, batch_size):
  train_loaders = []
  test_loaders = []
  
  for classes in task_splits:
    train_subset = select_class_subset(train_dataset, classes)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    test_subset = select_class_subset(test_dataset, classes)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)
    
  return train_loaders, test_loaders

def dataloaders(batch_size = 128, num_tasks=10):
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
  train_dataset = datasets.CIFAR100(root = "./data", train = True, download=True, transform=transform)
  test_dataset = datasets.CIFAR100(root = "./data", train = False, download=True, transform=transform)

  classes_per_task = 100 // num_tasks  # 예: 10
  task_splits = [list(range(i * classes_per_task, (i + 1) * classes_per_task)) for i in range(num_tasks)]
  
  train_loaders, test_loaders = create_task_dataloaders(train_dataset, test_dataset, task_splits, batch_size)
  
  return train_loaders, test_loaders

