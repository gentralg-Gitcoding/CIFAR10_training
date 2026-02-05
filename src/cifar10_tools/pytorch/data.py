'''Data loading and preprocessing functions for CIFAR-10 dataset.'''

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
):
    """
    Loads CIFAR-10, applies preprocessing with separate train/eval transforms,
    and returns DataLoaders.
    
    Args:
        data_dir: Path to CIFAR-10 data directory
        batch_size: Batch size for DataLoaders
        train_transform: Transform to apply to training data
        eval_transform: Transform to apply to validation and test data
        device: Device to preload tensors onto. If None, data stays on CPU 
                and transforms are applied on-the-fly during iteration.
        download: Whether to download the dataset if not present
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Load datasets with respective transforms
    train_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    
    val_test_dataset_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=eval_transform,
    )
    
    test_dataset = datasets.CIFAR10(
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

        # Train/val split (80/20)
        n_train = int(0.8 * len(X_train_full))
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
        # Train/val split (80/20) using Subset
        n_train = int(0.8 * len(train_dataset_full))
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


if __name__ == '__main__':

    download_cifar10_data()