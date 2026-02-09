Quick start guide
=================

This guide demonstrates common image classification tasks using the package.

Basic workflow
--------------

The typical workflow is:

1. Load and prepare your data
2. Define your model architecture
3. Train the model
4. Evaluate performance

Example: MNIST classification
------------------------------

This example shows the complete workflow using the MNIST dataset.

1. Load data
^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   import torch
   from torchvision import datasets, transforms
   from image_classification_tools.pytorch.data import (
       load_datasets, prepare_splits, create_dataloaders
   )

   # Define preprocessing
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])

   # Step 1: Load datasets
   train_dataset, test_dataset = load_datasets(
       data_source=datasets.MNIST,
       train_transform=transform,
       eval_transform=transform,
       download=True,
       root=Path('./data/mnist')
   )

   # Step 2: Prepare splits
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       train_val_split=0.8
   )

   # Step 3: Create dataloaders
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=True,
       device=device
   )

2. Define model
^^^^^^^^^^^^^^^

.. code-block:: python

   import torch.nn as nn

   model = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28 * 28, 512),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(512, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 10)
   )

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

3. Train model
^^^^^^^^^^^^^^

.. code-block:: python

   import torch.optim as optim
   from image_classification_tools.pytorch.training import train_model

   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       device=device,
       lazy_loading=False,  # Set to False when using preload_to_memory=True
       epochs=20,
       print_every=5
   )

4. Evaluate model
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from image_classification_tools.pytorch.evaluation import evaluate_model

   test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)
   print(f'Test accuracy: {test_accuracy:.2f}%')

5. Visualize results
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from image_classification_tools.pytorch.plotting import (
       plot_learning_curves, plot_confusion_matrix
   )

   # Learning curves
   fig, axes = plot_learning_curves(history)
   plt.show()

   # Confusion matrix
   class_names = [str(i) for i in range(10)]
   fig, ax = plot_confusion_matrix(true_labels, predictions, class_names)
   plt.show()

Working with custom datasets
-----------------------------

For datasets in ImageFolder format:

.. code-block:: python

   from pathlib import Path
   from torchvision.datasets import ImageFolder

   # Define transform
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
   ])

   # Load datasets from directory structure
   train_dataset, test_dataset = load_datasets(
       data_source=Path('./my_dataset'),
       train_transform=transform,
       eval_transform=transform
   )

   # If no test directory exists, use 3-way split
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset=train_dataset,
       test_dataset=test_dataset,  # Will be None if no test/ directory
       train_val_split=0.8,
       test_split=0.1  # Only used if test_dataset is None
   )

   # Create dataloaders
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=64,
       preload_to_memory=False,  # Lazy loading for large datasets
       num_workers=4
   )

Your directory structure should be:

.. code-block:: text

   my_dataset/
   ├── train/
   │   ├── class1/
   │   │   ├── img1.jpg
   │   │   └── img2.jpg
   │   └── class2/
   │       ├── img1.jpg
   │       └── img2.jpg
   └── test/
       ├── class1/
       └── class2/

Convolutional neural networks
------------------------------

For image data, CNNs typically perform better than fully connected networks:

.. code-block:: python

   # For 28x28 grayscale images (MNIST)
   model = nn.Sequential(
       nn.Conv2d(1, 32, kernel_size=3, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Conv2d(32, 64, kernel_size=3, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Flatten(),
       nn.Linear(64 * 7 * 7, 128),
       nn.ReLU(),
       nn.Linear(128, 10)
   ).to(device)

For color images (3 channels), change the first layer to ``nn.Conv2d(3, 32, ...)``.

Data augmentation
-----------------

Improve generalization with data augmentation:

.. code-block:: python

   # Training transform with augmentation
   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   # Evaluation transform (no augmentation)
   eval_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
   ])
   
   # Load with separate transforms
   train_dataset, test_dataset = load_datasets(
       data_source=datasets.MNIST,
       train_transform=train_transform,
       eval_transform=eval_transform,
       root=Path('./data/mnist')
   )
   
   # Prepare splits
   train_dataset, val_dataset, test_dataset = prepare_splits(
       train_dataset, test_dataset, train_val_split=0.8
   )
   
   # Create dataloaders with lazy loading (important for augmentation)
   train_loader, val_loader, test_loader = create_dataloaders(
       train_dataset, val_dataset, test_dataset,
       batch_size=128,
       preload_to_memory=False,  # Use lazy loading for on-the-fly augmentation
       num_workers=4,
       pin_memory=True
   )

Hyperparameter optimization
----------------------------

Use Optuna to find optimal hyperparameters:

.. code-block:: python

   import optuna
   from image_classification_tools.pytorch.hyperparameter_optimization import (
       create_objective
   )

   # Define search space
   search_space = {
       'batch_size': [32, 64, 128, 256],
       'n_conv_blocks': (1, 4),
       'initial_filters': [16, 32, 64],
       'n_fc_layers': (1, 3),
       'conv_dropout_rate': (0.1, 0.5),
       'fc_dropout_rate': (0.3, 0.7),
       'learning_rate': (1e-5, 1e-2, 'log'),
       'optimizer': ['Adam', 'SGD'],
       'weight_decay': (1e-6, 1e-3, 'log')
   }

   # Create objective function
   objective = create_objective(
       data_dir=Path('./data'),
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
   study.optimize(objective, n_trials=100)

   print(f'Best accuracy: {study.best_value:.2f}%')
   print(f'Best params: {study.best_params}')

Advanced: Building custom CNNs
-------------------------------

For more control, use the CNN builder:

.. code-block:: python

   from image_classification_tools.pytorch.hyperparameter_optimization import create_cnn

   model = create_cnn(
       n_conv_blocks=3,
       initial_filters=32,
       n_fc_layers=2,
       conv_dropout_rate=0.25,
       fc_dropout_rate=0.5,
       num_classes=10,
       in_channels=3
   ).to(device)

Next steps
----------

* See the :doc:`api/index` for detailed function documentation
* Check the `demo project <https://github.com/gperdrizet/CIFAR10>`_ for a complete CIFAR-10 example
