API reference
=============

The CIFAR-10 Tools package provides a comprehensive set of modules for building, training, 
and evaluating convolutional neural networks on the CIFAR-10 dataset.

Core modules
------------

.. toctree::
   :maxdepth: 2

   data
   training
   evaluation
   plotting
   hyperparameter_optimization

Module overview
---------------

**Data loading** (:mod:`image_classification_tools.pytorch.data`)
   Functions for loading and preprocessing CIFAR-10 data with flexible transforms and batching.

**Training** (:mod:`image_classification_tools.pytorch.training`)
   Utilities for model training with progress tracking and history logging.

**Evaluation** (:mod:`image_classification_tools.pytorch.evaluation`)
   Functions for evaluating model performance and generating predictions.

**Plotting** (:mod:`image_classification_tools.pytorch.plotting`)
   Visualization utilities for training curves, confusion matrices, and performance analysis.

**Hyperparameter optimization** (:mod:`image_classification_tools.pytorch.hyperparameter_optimization`)
   Optuna-based hyperparameter search with configurable architectures and search spaces.

Complete module index
---------------------

* :ref:`genindex`
* :ref:`modindex`
