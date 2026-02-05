Data loading
============

.. automodule:: cifar10_tools.pytorch.data
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: cifar10_tools.pytorch.data.make_data_loaders

Overview
--------

The data module provides flexible data loading capabilities for CIFAR-10 with support for:

* Custom train/eval transforms
* Configurable batch sizes
* Optional GPU preloading
* Automatic train/validation splitting (80/20)

Example usage
-------------

Basic data loading:

.. code-block:: python

   from pathlib import Path
   from torchvision import transforms
   from cifar10_tools.pytorch.data import make_data_loaders

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir=Path('./data/cifar10'),
       batch_size=128,
       train_transform=transform,
       eval_transform=transform
   )

With data augmentation:

.. code-block:: python

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

   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir=Path('./data/cifar10'),
       batch_size=128,
       train_transform=train_transform,
       eval_transform=eval_transform,
       device=None  # Keep on CPU for on-the-fly augmentation
   )
