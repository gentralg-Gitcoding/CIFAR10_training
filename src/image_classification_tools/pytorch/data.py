'''Data loading and preprocessing functions for image classification datasets.

This module provides utilities for loading datasets (including CIFAR-10) and creating
PyTorch DataLoaders with support for custom transforms and device preloading.
'''

import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def download_cifar10_data(data_dir: str='data/pytorch/cifar10'):
    '''Download CIFAR-10 dataset using torchvision.datasets.'''

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    _ = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True
    )

    _ = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True
    )


def make_data_loaders(
    data_dir: Path,
    batch_size: int,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    device: torch.device | None = None,
    download: bool = False,
    dataset_class = None,
    train_val_split: float = 0.8,
):
    """
    Loads image classification dataset, applies preprocessing with separate train/eval transforms,
    and returns DataLoaders.
    
    This function supports torchvision datasets (like CIFAR-10, CIFAR-100, MNIST) and can work
    with any dataset that follows the torchvision dataset interface.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for DataLoaders
        train_transform: Transform to apply to training data
        eval_transform: Transform to apply to validation and test data
        device: Device to preload tensors onto. If None, data stays on CPU 
                and transforms are applied on-the-fly during iteration.
        download: Whether to download the dataset if not present
        dataset_class: Dataset class to use (default: datasets.CIFAR10 for backward compatibility)
        train_val_split: Fraction of training data to use for training (default: 0.8)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Default to CIFAR10 for backward compatibility
    if dataset_class is None:
        dataset_class = datasets.CIFAR10

    # Load datasets with respective transforms
    train_dataset_full = dataset_class(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    val_test_dataset_full = dataset_class(
        root=data_dir,
        train=True,
        download=download,
        transform=eval_transform,
    )
    
    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )

    if device is not None:
        # Preload entire dataset to device for faster training
        X_train_full = torch.stack([img for img, _ in train_dataset_full]).to(device)
        y_train_full = torch.tensor([label for _, label in train_dataset_full]).to(device)
        
        X_val_test_full = torch.stack([img for img, _ in val_test_dataset_full]).to(device)
        y_val_test_full = torch.tensor([label for _, label in val_test_dataset_full]).to(device)

        X_test = torch.stack([img for img, _ in test_dataset]).to(device)
        y_test = torch.tensor([label for _, label in test_dataset]).to(device)

        # Train/val split using specified ratio
        n_train = int(train_val_split * len(X_train_full))
        indices = torch.randperm(len(X_train_full))

        X_train = X_train_full[indices[:n_train]]
        y_train = y_train_full[indices[:n_train]]
        X_val = X_val_test_full[indices[n_train:]]
        y_val = y_val_test_full[indices[n_train:]]

        # TensorDatasets
        train_tensor_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_tensor_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_tensor_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
    else:
        # Don't preload - use datasets directly for on-the-fly transforms
        # Train/val split using specified ratio
        n_train = int(train_val_split * len(train_dataset_full))
        indices = torch.randperm(len(train_dataset_full)).tolist()
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_tensor_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
        val_tensor_dataset = torch.utils.data.Subset(val_test_dataset_full, val_indices)
        test_tensor_dataset = test_dataset

    # DataLoaders
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def generate_augmented_data(X_train, y_train, augmentation_transforms, augmentations_per_image, 
                           save_path=None, force_reaugment=False):
    '''Generate augmented training data with optional saving and loading.
    
    Args:
        X_train: Training images tensor (on GPU or CPU)
        y_train: Training labels tensor
        augmentation_transforms: nn.Sequential containing augmentation transforms
        augmentations_per_image: Number of augmented versions to create per image
        save_path: Optional path to save/load augmented data
        force_reaugment: If True, regenerate even if saved data exists
    
    Returns:
        Tuple of (X_train_final, y_train_final) on CPU
    '''
    
    # Move data to CPU for augmentation
    X_train_cpu = X_train.cpu()
    y_train_cpu = y_train.cpu()
    
    # Check if saved augmented data exists
    if save_path and os.path.exists(save_path) and not force_reaugment:
        print(f'Loading pre-generated augmented data from {save_path}...')
        saved_data = torch.load(save_path)
        X_train_final = saved_data['X_train']
        y_train_final = saved_data['y_train']
        
        print(f'\nLoaded augmented training set:')
        print(f'  Total size: {len(X_train_final)}')
        print(f'  Original: {len(X_train_cpu)}')
        print(f'  Added: {len(X_train_final) - len(X_train_cpu)}')
        print(f'  Memory location: {X_train_final.device}')
        print(f'  Augmentation factor: {len(X_train_final) / len(X_train_cpu):.1f}x')
        
    else:
        if force_reaugment:
            print('Forcing re-augmentation...')
        else:
            print('No saved augmented data found. Generating augmentations...')
        
        # Lists to collect augmented data on CPU
        X_train_aug = [X_train_cpu]  # Start with original training data
        y_train_aug = [y_train_cpu]
        
        # Generate augmented versions on CPU
        # Apply transforms to each image individually to ensure independent transformations
        for aug_idx in range(augmentations_per_image):
            print(f'Creating augmentation batch {aug_idx + 1}/{augmentations_per_image}...')
            
            # Apply augmentations to each training image individually
            X_aug_list = []
            for img in X_train_cpu:
                # Add batch dimension, apply transform, remove batch dimension
                img_aug = augmentation_transforms(img.unsqueeze(0)).squeeze(0)
                X_aug_list.append(img_aug)
            
            X_aug = torch.stack(X_aug_list)
            
            X_train_aug.append(X_aug)
            y_train_aug.append(y_train_cpu)
        
        # Concatenate all training data (original + augmented) - stays on CPU
        X_train_final = torch.cat(X_train_aug, dim=0)
        y_train_final = torch.cat(y_train_aug, dim=0)
        
        print(f'\nAugmented training set size: {len(X_train_final)}')
        print(f'  Added: {len(X_train_final) - len(X_train_cpu)}')
        print(f'  Original: {len(X_train_cpu)}')
        print(f'  Memory location: {X_train_final.device}')
        print(f'  Augmentation factor: {len(X_train_final) / len(X_train_cpu):.1f}x')
        
        # Save augmented data for future use
        if save_path:
            print(f'\nSaving augmented data to {save_path}...')
            torch.save({
                'X_train': X_train_final,
                'y_train': y_train_final,
                'augmentations_per_image': augmentations_per_image,
                'original_train_size': len(X_train_cpu)
            }, save_path)
            print('Augmented data saved successfully!')
    
    return X_train_final, y_train_final


if __name__ == '__main__':

    download_cifar10_data()