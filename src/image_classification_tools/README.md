# Image Classification Tools

A lightweight PyTorch toolkit for building and training image classification models.

## Overview

This package provides utilities for common image classification tasks:

- **Data loading**: Flexible data loaders for torchvision datasets and custom image folders
- **Model training**: Training loops with progress tracking and validation
- **Evaluation**: Accuracy metrics, confusion matrices, and performance analysis
- **Visualization**: Learning curves, probability distributions, and evaluation plots
- **Hyperparameter optimization**: Optuna integration for automated model tuning

## Installation

```bash
pip install image-classification-tools
```

## Quick start

### Basic usage

```python
import torch
from torchvision import datasets, transforms
from image_classification_tools.pytorch.data import make_data_loaders
from image_classification_tools.pytorch.training import train_model
from image_classification_tools.pytorch.evaluation import evaluate_model

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create data loaders
train_loader, val_loader, test_loader = make_data_loaders(
    data_dir='./data',
    dataset_class=datasets.MNIST,
    batch_size=64,
    train_transform=transform,
    eval_transform=transform
)

# Define model, criterion, optimizer
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=10
)

# Evaluate
accuracy, predictions, labels = evaluate_model(model, test_loader)
print(f'Test accuracy: {accuracy:.2f}%')
```

### Hyperparameter optimization

```python
from image_classification_tools.pytorch.hyperparameter_optimization import create_objective
import optuna

# Define search space
search_space = {
    'batch_size': [32, 64, 128],
    'n_conv_blocks': (1, 3),
    'learning_rate': (1e-4, 1e-2, 'log'),
    'optimizer': ['Adam', 'SGD']
}

# Create objective function
objective = create_objective(
    data_dir='./data',
    train_transform=transform,
    eval_transform=transform,
    n_epochs=20,
    num_classes=10,
    input_size=(1, 28, 28),
    search_space=search_space
)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- numpy
- matplotlib
- optuna (optional, for hyperparameter optimization)

## Documentation

Full documentation is available at: https://gperdrizet.github.io/CIFAR10/

## Demo project

See a complete example of using this package for CIFAR-10 classification:
https://github.com/gperdrizet/CIFAR10

## License

GPLv3
