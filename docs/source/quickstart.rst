Quick start
===========

This guide will get you started with CIFAR-10 Tools in 5 minutes.

Basic training example
----------------------

Here's a minimal example of training a CNN on CIFAR-10:

.. code-block:: python

   from torchvision import transforms
   from image_classification_tools.pytorch.data import make_data_loaders
   from image_classification_tools.pytorch.hyperparameter_optimization import create_cnn
   from image_classification_tools.pytorch.training import train_model
   
   # Define input transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   # Load data
   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir='./data/cifar10',
       batch_size=128,
       train_transform=transform,
       eval_transform=transform,
       device='cuda'  # Preload to GPU for faster training
   )
   
   # Create model
   model = create_cnn(
       n_conv_blocks=3,
       initial_filters=32,
       n_fc_layers=2,
       base_kernel_size=3,
       conv_dropout_rate=0.2,
       fc_dropout_rate=0.5,
       pooling_strategy='max',
       use_batch_norm=True
   ).to('cuda')
   
   # Train
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       epochs=10,
       print_every=2
   )

Hyperparameter Optimization
----------------------------

Use Optuna to find the best hyperparameters:

.. code-block:: python

   import optuna
   from image_classification_tools.pytorch.hyperparameter_optimization import create_objective
   from torchvision import transforms

   # Define input transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Define search space
   search_space = {
       'batch_size': [64, 128, 256, 512],
       'n_conv_blocks': (2, 5),
       'initial_filters': [16, 32, 64],
       'n_fc_layers': (1, 4),
       'base_kernel_size': (3, 5),
       'conv_dropout_rate': (0.0, 0.5),
       'fc_dropout_rate': (0.2, 0.7),
       'pooling_strategy': ['max', 'avg'],
       'use_batch_norm': [True, False],
       'learning_rate': (1e-5, 1e-2, 'log'),
       'optimizer': ['Adam', 'SGD', 'RMSprop'],
       'sgd_momentum': (0.8, 0.99)
   }
   
   # Create objective function
   objective = create_objective(
       data_dir='./data/cifar10',
       train_transform=transform,
       eval_transform=transform,
       n_epochs=20,
       device='cuda',
       search_space=search_space
   )
   
   # Run optimization
   study = optuna.create_study(
       direction='maximize',
       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   )
   
   study.optimize(objective, n_trials=50, show_progress_bar=True)
   
   # Get best parameters
   print(f'Best validation accuracy: {study.best_trial.value:.2f}%')
   print('Best hyperparameters:')

   for key, value in study.best_trial.params.items():
       print(f"  {key}: {value}")

Data Augmentation
-----------------

Add data augmentation for improved generalization:

.. code-block:: python

   import optuna
   from image_classification_tools.pytorch.hyperparameter_optimization import create_objective
   from torchvision import transforms

   # Training transform with augmentation
   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(degrees=15),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   # Evaluation transform (no augmentation)
   eval_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   # Load with different transforms for train/eval
   train_loader, val_loader, test_loader = make_data_loaders(
       data_dir='./data/cifar10',
       batch_size=128,
       train_transform=train_transform,
       eval_transform=eval_transform,
       device=None  # Keep on CPU for on-the-fly augmentation
   )

Evaluation and Visualization
-----------------------------

Evaluate your model and visualize results:

.. code-block:: python

   from image_classification_tools.pytorch.evaluation import evaluate_model
   from image_classification_tools.pytorch.plotting import (
       plot_learning_curves,
       plot_confusion_matrix,
       plot_evaluation_curves
   )
   import matplotlib.pyplot as plt

   # Evaluate on test set
   test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)
   print(f'Test accuracy: {test_accuracy:.2f}%')
   
   # Plot learning curves
   fig, axes = plot_learning_curves(history)
   plt.show()
   
   # Plot confusion matrix
   class_names = [
       'airplane', 'automobile', 'bird', 'cat', 'deer',
       'dog', 'frog', 'horse', 'ship', 'truck'
   ]

   fig, ax = plot_confusion_matrix(true_labels, predictions, class_names)
   plt.show()

Next Steps
----------

* Explore the :doc:`notebooks/index` for detailed examples
* Check the :doc:`api/index` for complete API documentation
* See the `example notebooks <https://github.com/gperdrizet/CIFAR10/tree/main/notebooks>`_ on GitHub
