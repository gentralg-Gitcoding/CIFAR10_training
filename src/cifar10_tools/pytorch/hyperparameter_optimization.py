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


def create_cnn(
    n_conv_blocks: int,
    initial_filters: int,
    fc_units_1: int,
    fc_units_2: int,
    dropout_rate: float,
    use_batch_norm: bool,
    num_classes: int = 10,
    in_channels: int = 3,
    input_size: int = 32
) -> nn.Sequential:
    '''Create a CNN with configurable architecture.
    
    Args:
        n_conv_blocks: Number of convolutional blocks (1-5)
        initial_filters: Number of filters in first conv layer (doubles each block)
        fc_units_1: Number of units in first fully connected layer
        fc_units_2: Number of units in second fully connected layer
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        in_channels: Number of input channels (default: 3 for RGB)
        input_size: Input image size (default: 32 for CIFAR-10)
    
    Returns:
        nn.Sequential model
    '''
    layers = []
    current_channels = in_channels
    current_size = input_size
    
    for block_idx in range(n_conv_blocks):
        out_channels = initial_filters * (2 ** block_idx)
        
        # First conv in block
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())
        
        # Second conv in block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU())
        
        # Pooling and dropout
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.Dropout(dropout_rate))
        
        current_channels = out_channels
        current_size //= 2
    
    # Calculate flattened size
    final_channels = initial_filters * (2 ** (n_conv_blocks - 1))
    flattened_size = final_channels * current_size * current_size
    
    # Classifier (3 fully connected layers)
    layers.append(nn.Flatten())
    layers.append(nn.Linear(flattened_size, fc_units_1))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(fc_units_1, fc_units_2))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(fc_units_2, num_classes))
    
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
    num_classes: int = 10,
    in_channels: int = 3
) -> Callable[[optuna.Trial], float]:
    '''Create an Optuna objective function for CNN hyperparameter optimization.
    
    This factory function creates a closure that captures the data loaders and
    training configuration, returning an objective function suitable for Optuna.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of epochs per trial
        device: Device to train on (cuda or cpu)
        num_classes: Number of output classes (default: 10)
        in_channels: Number of input channels (default: 3 for RGB)
    
    Returns:
        Objective function for optuna.Study.optimize()
    
    Example:
        >>> objective = create_objective(train_loader, val_loader, n_epochs=50, device=device)
        >>> study = optuna.create_study(direction='maximize')
        >>> study.optimize(objective, n_trials=100)
    '''
    
    def objective(trial: optuna.Trial) -> float:
        '''Optuna objective function for CNN hyperparameter optimization.'''
        
        # Suggest hyperparameters
        n_conv_blocks = trial.suggest_int('n_conv_blocks', 1, 5)
        initial_filters = trial.suggest_categorical('initial_filters', [8, 16, 32, 64, 128])
        fc_units_1 = trial.suggest_categorical('fc_units_1', [128, 256, 512, 1024, 2048])
        fc_units_2 = trial.suggest_categorical('fc_units_2', [32, 64, 128, 256, 512])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.75)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        
        # Create model
        model = create_cnn(
            n_conv_blocks=n_conv_blocks,
            initial_filters=initial_filters,
            fc_units_1=fc_units_1,
            fc_units_2=fc_units_2,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            num_classes=num_classes,
            in_channels=in_channels
        ).to(device)
        
        # Define optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        elif optimizer_name == 'SGD':
            momentum = trial.suggest_float('sgd_momentum', 0.8, 0.99)
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

        except torch.cuda.OutOfMemoryError:
            # Clear CUDA cache and skip this trial
            torch.cuda.empty_cache()
            raise optuna.TrialPruned(f'CUDA OOM with params: {trial.params}')
    
    return objective
