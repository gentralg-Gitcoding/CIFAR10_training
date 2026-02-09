Data loading
============

.. automodule:: image_classification_tools.pytorch.data
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.data.load_datasets

.. autofunction:: image_classification_tools.pytorch.data.prepare_splits

.. autofunction:: image_classification_tools.pytorch.data.create_dataloaders

.. autofunction:: image_classification_tools.pytorch.data.generate_augmented_data

Overview
--------

The data module provides a flexible three-step data loading workflow:

1. **Load datasets**: Load train/test datasets from PyTorch dataset classes or directories
2. **Prepare splits**: Split data into train/val(/test) with configurable ratios
3. **Create dataloaders**: Create DataLoaders with optional memory preloading strategies

Key features:

* Support for torchvision datasets (CIFAR-10, MNIST, etc.) and custom ImageFolder datasets
* Separate train and evaluation transforms
* Flexible splitting: 2-way (train/val) or 3-way (train/val/test)
* Three memory strategies: lazy loading, CPU preloading, or GPU preloading
* Data augmentation with chunking for large datasets
* Configurable batch sizes and workers

Example usage
-------------

Basic workflow (CIFAR-10 with GPU preloading):

.. code-block:: python

   from pathlib import Path
   import torch
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.data import (
       load_datasets, prepare_splits, create_dataloaders
   )

   # Define transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Step 1: Load datasets
   train_dataset, test_dataset = load_datasets(
       data_source=datasets.CIFAR10,
       train_transform=transform,
       eval_transform=transform,
       download=True,
       root=Path('./data/cifar10')
   )

   # Step 2: Prepare splits (2-way: train/val from train_dataset)
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       train_val_split=0.8  # 80% train, 20% val
   )

   # Step 3: Create dataloaders with GPU preloading
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=True,
       device=device
   )

With data augmentation (lazy loading):

.. code-block:: python

   # Define separate transforms for training and evaluation
   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   eval_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Load with different transforms
   train_dataset, test_dataset = load_datasets(
       data_source=datasets.CIFAR10,
       train_transform=train_transform,
       eval_transform=eval_transform,
       root=Path('./data/cifar10')
   )

   # Prepare splits
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       train_val_split=0.8
   )

   # Create dataloaders with lazy loading (no preloading)
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=False,  # Lazy loading for augmentation
       num_workers=4,
       pin_memory=True
   )

3-way split (no separate test set):

.. code-block:: python

   # Load only training data (no test set available)
   train_dataset, _ = load_datasets(
       data_source=datasets.ImageFolder,
       train_transform=transform,
       eval_transform=transform,
       root=Path('./my_dataset/train')
   )

   # 3-way split: train/val/test all from train_dataset
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=None,  # Will split test from train_dataset
       train_val_split=0.8,  # 80/20 split of remaining data
       test_split=0.15  # Reserve 15% for testing
   )
   # Results in approximately: 68% train, 17% val, 15% test

