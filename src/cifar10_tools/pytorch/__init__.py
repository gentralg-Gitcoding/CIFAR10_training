'''PyTorch utilities for CIFAR-10 classification.'''

from cifar10_tools.pytorch.data import load_cifar10
from cifar10_tools.pytorch.evaluation import evaluate_model
from cifar10_tools.pytorch.training import train_model
from cifar10_tools.pytorch.plotting import (
    plot_sample_images,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_class_probability_distributions,
    plot_evaluation_curves,
    plot_optimization_results
)
from cifar10_tools.pytorch.hyperparameter_optimization import (
    create_cnn,
    train_trial,
    create_objective
)

__all__ = [
    'load_cifar10',
    'evaluate_model',
    'train_model',
    'plot_sample_images',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_class_probability_distributions',
    'plot_evaluation_curves',
    'plot_optimization_results',
    'create_cnn',
    'train_trial',
    'create_objective'
]
