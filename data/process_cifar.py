import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_data(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def get_cifar10_stats():
    """Get dataset statistics."""
    return {
        'num_classes': 10,
        'image_shape': (3, 32, 32),
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5)
    }