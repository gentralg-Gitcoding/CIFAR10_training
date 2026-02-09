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
from pathlib import Path
from torchvision import datasets, transforms
from image_classification_tools.pytorch.data import (
    load_datasets, prepare_splits, create_dataloaders
)
from image_classification_tools.pytorch.training import train_model
from image_classification_tools.pytorch.evaluation import evaluate_model

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset, test_dataset = load_datasets(
    data_source=datasets.MNIST,
    train_transform=transform,
    eval_transform=transform,
    download=True,
    root=Path('./data/mnist')
)

# Prepare splits
train_dataset, val_dataset, test_dataset = prepare_splits(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_val_split=0.8
)

# Create dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=64,
    preload_to_memory=True,
    device=device
)

# Define model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train with lazy loading (moves batches to device during training)
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    lazy_loading=True,  # Set False if data already on device
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
    'initial_filters': [16, 32, 64],
    'n_fc_layers': (1, 3),
    'conv_dropout_rate': (0.1, 0.5),
    'fc_dropout_rate': (0.3, 0.7),
    'learning_rate': (1e-4, 1e-2, 'log'),
    'optimizer': ['Adam', 'SGD'],
    'weight_decay': (1e-6, 1e-3, 'log')
}

# Create objective function
objective = create_objective(
    data_dir='./data',
    train_transform=transform,
    eval_transform=transform,
    n_epochs=20,
    device=device,
    num_classes=10,
    in_channels=1,
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
