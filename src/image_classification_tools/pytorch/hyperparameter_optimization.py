'''Hyperparameter optimization utilities for CNN models using Optuna.

This module provides functions for building configurable CNN architectures
and running hyperparameter optimization with Optuna.
'''

from typing import Callable

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from image_classification_tools.pytorch.data import make_data_loaders


def create_cnn(
    n_conv_blocks: int,
    initial_filters: int,
    n_fc_layers: int,
    base_kernel_size: int,
    conv_dropout_rate: float,
    fc_dropout_rate: float,
    pooling_strategy: str,
    use_batch_norm: bool,
    num_classes: int,
    in_channels: int = 3,
    input_size: int = 32
) -> nn.Sequential:
    '''Create a CNN with configurable architecture.
    
    This function builds a flexible CNN architecture suitable for various image 
    classification tasks. The architecture is parameterized to work with different
    input sizes, channel counts, and number of output classes.
    
    Args:
        n_conv_blocks: Number of convolutional blocks (1-5)
        initial_filters: Number of filters in first conv layer (doubles each block)
        n_fc_layers: Number of fully connected layers (1-8)
        base_kernel_size: Base kernel size (decreases by 2 per block, min 3)
        conv_dropout_rate: Dropout probability after convolutional blocks
        fc_dropout_rate: Dropout probability in fully connected layers
        pooling_strategy: Pooling type ('max' or 'avg')
        use_batch_norm: Whether to use batch normalization
        num_classes: Number of output classes (required)
        in_channels: Number of input channels (default: 3 for RGB images)
        input_size: Input image size in pixels (default: 32, e.g., for CIFAR-10)
    
    Returns:
        nn.Sequential model
    '''

    layers = []
    current_channels = in_channels
    current_size = input_size
    
    # Convolutional blocks
    for block_idx in range(n_conv_blocks):
        out_channels = initial_filters * (2 ** block_idx)
        kernel_size = max(3, base_kernel_size - 2 * block_idx)
        padding = kernel_size // 2
        
        # First conv in block
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, padding=padding))
        # Update size after conv: output_size = (input_size + 2*padding - kernel_size) + 1
        current_size = (current_size + 2 * padding - kernel_size) + 1

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())
        
        # Second conv in block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        current_size = (current_size + 2 * padding - kernel_size) + 1

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())
        
        # Pooling
        if pooling_strategy == 'max':
            layers.append(nn.MaxPool2d(2, 2))
        else:  # avg
            layers.append(nn.AvgPool2d(2, 2))
        
        layers.append(nn.Dropout(conv_dropout_rate))
        
        current_channels = out_channels
        current_size //= 2  # Pooling halves the size
    
    # Calculate flattened size using actual current_size
    flattened_size = current_channels * current_size * current_size
    
    # Classifier - dynamic FC layers with halving pattern
    layers.append(nn.Flatten())
    
    # Generate FC layer sizes by halving from flattened_size
    fc_sizes = []
    current_fc_size = flattened_size // 2
    for _ in range(n_fc_layers):
        fc_sizes.append(max(10, current_fc_size))  # Minimum 10 units
        current_fc_size //= 2
    
    # Add FC layers
    in_features = flattened_size
    for fc_size in fc_sizes:
        layers.append(nn.Linear(in_features, fc_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(fc_dropout_rate))
        in_features = fc_size
    
    # Output layer
    layers.append(nn.Linear(in_features, num_classes))
    
    return nn.Sequential(*layers)


def train_trial(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    trial: optuna.Trial
) -> float:
    '''Train a model for a single Optuna trial with pruning support.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer for training
        criterion: Loss function
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of epochs to train
        trial: Optuna trial object for reporting and pruning
    
    Returns:
        Best validation accuracy achieved during training
    '''
    best_val_accuracy = 0.0
    
    for epoch in range(n_epochs):

        # Training phase
        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        
        # Report intermediate value for pruning
        trial.report(val_accuracy, epoch)
        
        # Prune unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_accuracy


def create_objective(
    data_dir,
    train_transform,
    eval_transform,
    n_epochs: int,
    device: torch.device,
    num_classes: int,
    in_channels: int = 3,
    input_size: int = 32,
    search_space: dict = None
) -> Callable[[optuna.Trial], float]:
    '''Create an Optuna objective function for CNN hyperparameter optimization.
    
    This factory function creates a closure that captures the data loading parameters
    and training configuration, returning an objective function suitable for Optuna.
    The function is dataset-agnostic and works with any image classification task.
    
    Args:
        data_dir: Directory containing training data
        train_transform: Transform to apply to training data
        eval_transform: Transform to apply to validation data
        n_epochs: Number of epochs per trial
        device: Device to train on (cuda or cpu)
        num_classes: Number of output classes (required, e.g., 10 for CIFAR-10, 1000 for ImageNet)
        in_channels: Number of input channels (default: 3 for RGB images, 1 for grayscale)
        input_size: Input image size in pixels (default: 32, adjust for your dataset)
        search_space: Dictionary defining hyperparameter search space (default: None)
    
    Returns:
        Objective function for optuna.Study.optimize()
    
    Example:
        >>> objective = create_objective(
        ...     data_dir='data/', 
        ...     train_transform=transform, 
        ...     eval_transform=transform,
        ...     n_epochs=50, 
        ...     device=device,
        ...     num_classes=10,
        ...     input_size=32
        ... )
        >>> study = optuna.create_study(direction='maximize')
        >>> study.optimize(objective, n_trials=100)
    '''
    
    # Default search space if none provided
    if search_space is None:
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
    
    def objective(trial: optuna.Trial) -> float:
        '''Optuna objective function for CNN hyperparameter optimization.'''
        
        # Suggest hyperparameters from search space
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_size'])
        n_conv_blocks = trial.suggest_int('n_conv_blocks', *search_space['n_conv_blocks'])
        initial_filters = trial.suggest_categorical('initial_filters', search_space['initial_filters'])
        n_fc_layers = trial.suggest_int('n_fc_layers', *search_space['n_fc_layers'])
        base_kernel_size = trial.suggest_int('base_kernel_size', *search_space['base_kernel_size'])
        conv_dropout_rate = trial.suggest_float('conv_dropout_rate', *search_space['conv_dropout_rate'])
        fc_dropout_rate = trial.suggest_float('fc_dropout_rate', *search_space['fc_dropout_rate'])
        pooling_strategy = trial.suggest_categorical('pooling_strategy', search_space['pooling_strategy'])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', search_space['use_batch_norm'])
        
        # Handle learning rate with optional log scale
        lr_params = search_space['learning_rate']
        learning_rate = trial.suggest_float('learning_rate', lr_params[0], lr_params[1], 
                                           log=(lr_params[2] == 'log' if len(lr_params) > 2 else False))
        
        optimizer_name = trial.suggest_categorical('optimizer', search_space['optimizer'])
        
        # Create data loaders with suggested batch size
        train_loader, val_loader, _ = make_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            device=device,
            download=False
        )
        
        # Create model
        model = create_cnn(
            n_conv_blocks=n_conv_blocks,
            initial_filters=initial_filters,
            n_fc_layers=n_fc_layers,
            base_kernel_size=base_kernel_size,
            conv_dropout_rate=conv_dropout_rate,
            fc_dropout_rate=fc_dropout_rate,
            pooling_strategy=pooling_strategy,
            use_batch_norm=use_batch_norm,
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size
        ).to(device)
        
        # Define optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('sgd_momentum', *search_space['sgd_momentum'])
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
        else:  # RMSprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        # Train model and return best validation accuracy
        try:
            return train_trial(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                trial=trial
            )

        except RuntimeError as e:
            # Catch architecture errors (e.g., dimension mismatches)
            raise optuna.TrialPruned(f'RuntimeError with params: {trial.params} - {str(e)}')
        
        except torch.cuda.OutOfMemoryError:
            # Clear CUDA cache and skip this trial
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f'CUDA OOM with params: {trial.params}')
    
    return objective
