Data loading
============

.. automodule:: image_classification_tools.pytorch.data
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.data.make_data_loaders

Overview
--------

The data module provides flexible data loading capabilities with support for:

* torchvision datasets (CIFAR-10, MNIST, ImageFolder, etc.)
* Custom train/eval transforms
* Configurable batch sizes
* Optional GPU preloading
* Automatic train/validation splitting (default 80/20)

Example usage
-------------

MNIST dataset:

.. code-block:: python

   from pathlib import Path
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.data import make_data_loaders

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir=Path('./data'),
       dataset_class=datasets.MNIST,
       batch_size=128,
       train_transform=transform,
       eval_transform=transform
   )

CIFAR-10 dataset:

.. code-block:: python

   from torchvision import datasets

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir=Path('./data'),
       dataset_class=datasets.CIFAR10,
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
       data_dir=Path('./data'),
       dataset_class=datasets.CIFAR10,
       batch_size=128,
       train_transform=train_transform,
       eval_transform=eval_transform,
       device=None  # Keep on CPU for on-the-fly augmentation
   )

Custom dataset with ImageFolder:

.. code-block:: python

   from torchvision.datasets import ImageFolder

   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir=Path('./my_dataset'),
       dataset_class=ImageFolder,
       batch_size=64,
       train_transform=transform,
       eval_transform=transform
   )
