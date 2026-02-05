Hyperparameter optimization
============================

.. automodule:: cifar10_tools.pytorch.hyperparameter_optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The hyperparameter optimization module provides Optuna integration for automated CNN architecture 
search and hyperparameter tuning. It includes:

* **Dynamic CNN architecture** with configurable depth, width, and components
* **Flexible search spaces** defined via dictionaries
* **Automatic trial pruning** for faster optimization
* **Error handling** for OOM and architecture mismatches

Key components
--------------

create_cnn
~~~~~~~~~~

Creates a CNN with dynamic architecture based on hyperparameters:

* Variable number of convolutional blocks (1-5)
* Doubling filter sizes per block
* Configurable kernel sizes (decreasing pattern)
* Dynamic fully-connected layers with halving sizes
* Separate dropout for conv and FC layers
* Choice of max or average pooling
* Optional batch normalization

create_objective
~~~~~~~~~~~~~~~~

Factory function that creates an Optuna objective for hyperparameter search:

* Accepts configurable search space dictionary
* Creates data loaders per trial with suggested batch size
* Handles architecture errors gracefully
* Supports MedianPruner for early stopping

train_trial
~~~~~~~~~~~

Trains a model for a single Optuna trial with pruning support.

Example usage
-------------

Basic hyperparameter optimization:

.. code-block:: python

   import optuna
   from cifar10_tools.pytorch.hyperparameter_optimization import create_objective

   # Define search space
   search_space = {
       'batch_size': [64, 128, 256, 512, 1024],
       'n_conv_blocks': (1, 5),
       'initial_filters': [8, 16, 32, 64, 128],
       'n_fc_layers': (1, 8),
       'base_kernel_size': (3, 7),
       'conv_dropout_rate': (0.0, 0.5),
       'fc_dropout_rate': (0.2, 0.75),
       'pooling_strategy': ['max', 'avg'],
       'use_batch_norm': [True, False],
       'learning_rate': (1e-5, 1e-1, 'log'),
       'optimizer': ['Adam', 'SGD', 'RMSprop'],
       'sgd_momentum': (0.8, 0.99)
   }

   # Create objective
   objective = create_objective(
       data_dir='./data/cifar10',
       train_transform=transform,
       eval_transform=transform,
       n_epochs=50,
       device=device,
       search_space=search_space
   )

   # Run optimization
   study = optuna.create_study(
       direction='maximize',
       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   )
   
   study.optimize(objective, n_trials=100, show_progress_bar=True)

   # Get best model
   best_params = study.best_trial.params
   print(f"Best accuracy: {study.best_trial.value:.2f}%")

With persistent storage:

.. code-block:: python

   from pathlib import Path

   # SQLite storage for resumable studies
   storage_path = Path('./optimization.db')
   storage_url = f'sqlite:///{storage_path}'

   study = optuna.create_study(
       study_name='cnn_optimization',
       direction='maximize',
       storage=storage_url,
       load_if_exists=True,  # Resume if interrupted
       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
   )

   study.optimize(objective, n_trials=200)

Creating the final model:

.. code-block:: python

   from cifar10_tools.pytorch.hyperparameter_optimization import create_cnn

   best_params = study.best_trial.params

   model = create_cnn(
       n_conv_blocks=best_params['n_conv_blocks'],
       initial_filters=best_params['initial_filters'],
       n_fc_layers=best_params['n_fc_layers'],
       base_kernel_size=best_params['base_kernel_size'],
       conv_dropout_rate=best_params['conv_dropout_rate'],
       fc_dropout_rate=best_params['fc_dropout_rate'],
       pooling_strategy=best_params['pooling_strategy'],
       use_batch_norm=best_params['use_batch_norm']
   ).to(device)

Search space format
-------------------

The search space dictionary supports three formats:

* **List**: Categorical choices - ``[64, 128, 256]``
* **Tuple (2 elements)**: Continuous range - ``(0.0, 0.5)`` for float, ``(1, 8)`` for int
* **Tuple (3 elements)**: Range with scale - ``(1e-5, 1e-1, 'log')`` for log-scaled float

Default search space includes:

* Batch size: [64, 128, 256, 512, 1024]
* Conv blocks: 1-5
* Initial filters: [8, 16, 32, 64, 128]
* FC layers: 1-8
* Kernel size: 3-7
* Conv dropout: 0.0-0.5
* FC dropout: 0.2-0.75
* Pooling: ['max', 'avg']
* Batch norm: [True, False]
* Learning rate: 1e-5 to 1e-1 (log scale)
* Optimizer: ['Adam', 'SGD', 'RMSprop']
* SGD momentum: 0.8-0.99
