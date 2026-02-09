'''Data loading and preprocessing functions for image classification datasets.

This module provides utilities for loading datasets (including CIFAR-10) and creating
PyTorch DataLoaders with support for custom transforms and device preloading.
'''

import os
from pathlib import Path
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset


def load_datasets(
    data_source: Path | type,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    download: bool = False,
    **dataset_kwargs
) -> Tuple[Dataset, Dataset | None]:
    '''Load training and test datasets from a directory or PyTorch dataset class.
    
    This function provides a flexible interface for loading image classification datasets.
    It supports both PyTorch built-in datasets (CIFAR-10, CIFAR-100, MNIST, etc.) and
    custom datasets stored in directories following the ImageFolder structure.
    
    Args:
        data_source: Either a Path to a directory containing train/test subdirectories,
                    or a PyTorch dataset class (e.g., datasets.CIFAR10)
        train_transform: Transforms to apply to training data
        eval_transform: Transforms to apply to test (and later validation) data
        download: Whether to download the dataset if using a PyTorch dataset class.
                 Ignored for directory-based datasets.
        **dataset_kwargs: Additional keyword arguments passed to the dataset class
                         (e.g., root='data/pytorch/cifar10')
    
    Returns:
        Tuple of (train_dataset, test_dataset). If the dataset doesn't have a separate
        test set (like ImageFolder), test_dataset will be None.
    '''
    
    if isinstance(data_source, Path):

        # Directory-based dataset using ImageFolder
        train_dir = data_source / 'train'
        test_dir = data_source / 'test'
        
        if not train_dir.exists():
            raise ValueError(f'Training directory not found: {train_dir}')
        
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=train_transform
        )
        
        # Test directory is optional
        test_dataset = None

        if test_dir.exists():
            test_dataset = datasets.ImageFolder(
                root=test_dir,
                transform=eval_transform
            )
    
    else:

        # PyTorch dataset class (CIFAR-10, MNIST, etc.)
        dataset_class = data_source
        
        train_dataset = dataset_class(
            train=True,
            download=download,
            transform=train_transform,
            **dataset_kwargs
        )
        
        test_dataset = dataset_class(
            train=False,
            download=download,
            transform=eval_transform,
            **dataset_kwargs
        )
    
    return train_dataset, test_dataset


def prepare_splits(
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    train_val_split: float = 0.8,
    test_split: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    '''Split training dataset into train/val(/test) splits.
    
    The splitting behavior depends on whether a separate test dataset is provided:
    - If test_dataset is provided: Split train_dataset into train/val only (2-way split)
    - If test_dataset is None: Split train_dataset into train/val/test (3-way split)
    
    Args:
        train_dataset: Training dataset to split
        test_dataset: Test dataset. If None, test set will be split from train_dataset.
        train_val_split: Fraction of (remaining) data to use for training vs validation.
                        - With test_dataset: train=80%, val=20% of train_dataset
                        - Without test_dataset: train=80%, val=20% of (train_dataset - test portion)
        test_split: Fraction of train_dataset to reserve for testing when test_dataset is None.
                   Only used when test_dataset is None. Default: 0.1 (10%)
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    
    Examples:
        # 2-way split: Pass separate test set
        train_ds, val_ds, test_ds = prepare_splits(
            train_dataset=my_train_data,
            test_dataset=my_test_data,  # Use this for testing
            train_val_split=0.8  # 80% train, 20% val
        )
        
        # 3-way split: No separate test set
        train_ds, val_ds, test_ds = prepare_splits(
            train_dataset=my_full_data,
            test_dataset=None,  # Will split test from train_dataset
            train_val_split=0.8,  # 80/20 split of remaining data after test
            test_split=0.15  # Reserve 15% for testing
        )
        # Results in: ~68% train, ~17% val, ~15% test
    '''
    
    if test_dataset is not None:

        # 2-way split: train/val only, use provided test set
        n_train = int(train_val_split * len(train_dataset))
        indices = torch.randperm(len(train_dataset)).tolist()
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset_final = Subset(train_dataset, train_indices)
        val_dataset_final = Subset(train_dataset, val_indices)
        test_dataset_final = test_dataset
        
    else:

        # 3-way split: train/val/test all from train_dataset
        total_size = len(train_dataset)
        n_test = int(test_split * total_size)
        n_train_val = total_size - n_test
        n_train = int(train_val_split * n_train_val)
        
        indices = torch.randperm(total_size).tolist()
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train_val]
        test_indices = indices[n_train_val:]
        
        train_dataset_final = Subset(train_dataset, train_indices)
        val_dataset_final = Subset(train_dataset, val_indices)
        test_dataset_final = Subset(train_dataset, test_indices)
    
    return train_dataset_final, val_dataset_final, test_dataset_final


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    preload_to_memory: bool = False,
    device: torch.device | None = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Create DataLoaders from prepared datasets with optional memory preloading.
    
    This function provides three memory management strategies:
    1. Lazy loading (preload_to_memory=False): Data stays on disk, loaded per batch
    2. CPU preloading (preload_to_memory=True, device=cpu): Entire dataset in RAM
    3. GPU preloading (preload_to_memory=True, device=cuda): Entire dataset in VRAM
    
    Args:
        train_dataset: Prepared training dataset
        val_dataset: Prepared validation dataset
        test_dataset: Prepared test dataset
        batch_size: Batch size for all DataLoaders
        shuffle_train: Whether to shuffle training data (default: True)
        num_workers: Number of subprocesses for data loading (default: 0 for single process).
                    Note: num_workers is ignored when preload_to_memory=True.
        preload_to_memory: If True, convert datasets to tensors and load into memory.
                          If False, keep as lazy-loading Dataset objects (default: False).
        device: Device to preload tensors onto. Only used if preload_to_memory=True.
               If None with preload_to_memory=True, defaults to CPU.
               Common values: torch.device('cpu'), torch.device('cuda')
        **kwargs: Additional keyword arguments passed to DataLoader
                 (e.g., pin_memory=True, persistent_workers=True)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Examples:
        # Strategy 1: Lazy loading (large datasets)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            num_workers=4,
            pin_memory=True
        )
        
        # Strategy 2: CPU preloading (medium datasets)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            preload_to_memory=True,
            device=torch.device('cpu')
        )
        
        # Strategy 3: GPU preloading (small datasets, fastest training)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=128,
            preload_to_memory=True,
            device=torch.device('cuda')
        )
    '''
    
    if preload_to_memory:
    
        # Preload datasets to memory
        if device is None:
            device = torch.device('cpu')
        
        # Load train data
        X_train = torch.stack([img for img, _ in train_dataset]).to(device)
        y_train = torch.tensor([label for _, label in train_dataset]).to(device)
        train_dataset_final = TensorDataset(X_train, y_train)
        
        # Load val data
        X_val = torch.stack([img for img, _ in val_dataset]).to(device)
        y_val = torch.tensor([label for _, label in val_dataset]).to(device)
        val_dataset_final = TensorDataset(X_val, y_val)
        
        # Load test data
        X_test = torch.stack([img for img, _ in test_dataset]).to(device)
        y_test = torch.tensor([label for _, label in test_dataset]).to(device)
        test_dataset_final = TensorDataset(X_test, y_test)
        
        # When preloading, num_workers should be 0
        num_workers = 0

    else:

        # Use datasets as-is for lazy loading
        train_dataset_final = train_dataset
        val_dataset_final = val_dataset
        test_dataset_final = test_dataset
    
    train_loader = DataLoader(
        train_dataset_final,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )
    
    return train_loader, val_loader, test_loader


def generate_augmented_data(
    train_dataset: Dataset,
    augmentation_transforms: torch.nn.Sequential,
    augmentations_per_image: int,
    save_path: str | Path | None = None,
    force_reaugment: bool = False,
    chunk_size: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Generate augmented training data from a dataset with optional chunking for large datasets.
    
    This function applies augmentation transforms to create multiple augmented versions of each
    training image. For large datasets, chunking prevents out-of-memory errors by processing
    and saving data incrementally.
    
    Args:
        train_dataset: PyTorch Dataset containing training images
        augmentation_transforms: nn.Sequential containing augmentation transforms to apply
        augmentations_per_image: Number of augmented versions to create per image
        save_path: Optional path to save/load augmented data. Recommended for large datasets.
        force_reaugment: If True, regenerate even if saved data exists
        chunk_size: Number of images to process per chunk. If None, processes entire dataset
                   at once. Use smaller values (e.g., 5000-10000) for datasets that don't
                   fit in memory. Ignored when loading from save_path.
    
    Returns:
        Tuple of (X_train_final, y_train_final) tensors on CPU containing original + augmented data
    
    Examples:
        # Small dataset - process all at once
        X_aug, y_aug = generate_augmented_data(
            train_dataset,
            augmentation_transforms,
            augmentations_per_image=2
        )
        
        # Large dataset - use chunking and save to disk
        X_aug, y_aug = generate_augmented_data(
            train_dataset,
            augmentation_transforms,
            augmentations_per_image=2,
            save_path='data/augmented/my_dataset.pt',
            chunk_size=5000
        )
    '''
    
    # Check if saved augmented data exists
    if save_path and os.path.exists(save_path) and not force_reaugment:
        print(f'Loading pre-generated augmented data from {save_path}...')
        saved_data = torch.load(save_path)
        X_train_final = saved_data['X_train']
        y_train_final = saved_data['y_train']
        
        print(f'\nLoaded augmented training set:')
        print(f'  Total size: {len(X_train_final)}')
        print(f'  Original: {saved_data.get("original_train_size", "unknown")}')
        print(f'  Added: {len(X_train_final) - saved_data.get("original_train_size", 0)}')
        print(f'  Memory location: {X_train_final.device}')
        print(f'  Augmentation factor: {saved_data.get("augmentations_per_image", "unknown") + 1}x')
        
        return X_train_final, y_train_final
    
    # Generate augmented data
    if force_reaugment:
        print('Forcing re-augmentation...')
    else:
        print('No saved augmented data found. Generating augmentations...')
    
    original_size = len(train_dataset)
    
    # Determine chunking strategy
    if chunk_size is None or chunk_size >= original_size:
        # Process entire dataset at once
        chunk_size = original_size
        num_chunks = 1
    else:
        num_chunks = (original_size + chunk_size - 1) // chunk_size
    
    print(f'Processing {original_size} images in {num_chunks} chunk(s) of size {chunk_size}')
    
    # Lists to collect all data
    all_images = []
    all_labels = []
    
    # Process dataset in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, original_size)
        chunk_range = range(start_idx, end_idx)
        
        print(f'\nProcessing chunk {chunk_idx + 1}/{num_chunks} (images {start_idx}-{end_idx-1})...')
        
        # Extract chunk data
        chunk_images = []
        chunk_labels = []
        for idx in chunk_range:
            img, label = train_dataset[idx]
            chunk_images.append(img)
            chunk_labels.append(label)
        
        X_chunk = torch.stack(chunk_images).cpu()
        y_chunk = torch.tensor(chunk_labels).cpu()
        
        # Start with original chunk data
        chunk_all_images = [X_chunk]
        chunk_all_labels = [y_chunk]
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            print(f'  Creating augmentation {aug_idx + 1}/{augmentations_per_image}...')
            
            # Apply augmentations to each image individually
            X_aug_list = []
            for img in X_chunk:
                # Add batch dimension, apply transform, remove batch dimension
                img_aug = augmentation_transforms(img.unsqueeze(0)).squeeze(0)
                X_aug_list.append(img_aug)
            
            X_aug = torch.stack(X_aug_list)
            chunk_all_images.append(X_aug)
            chunk_all_labels.append(y_chunk)
        
        # Concatenate chunk data
        X_chunk_final = torch.cat(chunk_all_images, dim=0)
        y_chunk_final = torch.cat(chunk_all_labels, dim=0)
        
        all_images.append(X_chunk_final)
        all_labels.append(y_chunk_final)
        
        print(f'  Chunk {chunk_idx + 1} complete: {len(X_chunk_final)} images')
    
    # Concatenate all chunks
    X_train_final = torch.cat(all_images, dim=0)
    y_train_final = torch.cat(all_labels, dim=0)
    
    print(f'\nAugmented training set complete:')
    print(f'  Total size: {len(X_train_final)}')
    print(f'  Original: {original_size}')
    print(f'  Added: {len(X_train_final) - original_size}')
    print(f'  Memory location: {X_train_final.device}')
    print(f'  Augmentation factor: {len(X_train_final) / original_size:.1f}x')
    
    # Save augmented data for future use
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f'\nSaving augmented data to {save_path}...')
        torch.save({
            'X_train': X_train_final,
            'y_train': y_train_final,
            'augmentations_per_image': augmentations_per_image,
            'original_train_size': original_size
        }, save_path)
        print('Augmented data saved successfully!')
    
    return X_train_final, y_train_final
