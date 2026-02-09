CIFAR-10 Tools Documentation
=============================

**CIFAR-10 Tools** is a comprehensive PyTorch toolkit for building, training, and optimizing 
convolutional neural networks (CNNs) for image classification on the CIFAR-10 dataset.

The package provides modular, reusable components for:

* **Data loading** with flexible preprocessing and augmentation
* **Model training** with configurable architectures
* **Hyperparameter optimization** using Optuna
* **Evaluation and visualization** of model performance

Key features
------------

**Modular design**
   Clean, composable functions for every stage of the ML pipeline

**Hyperparameter optimization**
   Built-in Optuna integration with configurable search spaces

**Rich visualizations**
   Comprehensive plotting utilities for training curves, confusion matrices, and performance analysis

Quick example
-------------

.. code-block:: python

   import optuna
   from image_classification_tools.pytorch.hyperparameter_optimization import create_objective
   from torchvision import transforms

   # Define input data transforms
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])

   # Define hyperparameter search space
   search_space = {
       'batch_size': [64, 128, 256, 512, 1024],
       'learning_rate': (1e-5, 1e-1, 'log'),
       'n_conv_blocks': [1, 2, 3, 4, 5]
   }

   # Create and optimize model
   objective = create_objective(
       data_dir='./data',
       train_transform=transform,
       eval_transform=transform,
       n_epochs=50,
       device='cuda',
       search_space=search_space
   )
   
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)

Getting started
---------------

To run the demo notebooks, clone the repository and use the provided devcontainer environment. To use the ``image_classification_tools`` package in your own projects, install from PyPI:

.. code-block:: bash
    
   pip install image_classification_tools

See :doc:`installation` for detailed setup instructions.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project links

   GitHub Repository <https://github.com/gperdrizet/CIFAR10>
   PyPI Package <https://pypi.org/project/image_classification_tools>
   Issue Tracker <https://github.com/gperdrizet/CIFAR10/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
