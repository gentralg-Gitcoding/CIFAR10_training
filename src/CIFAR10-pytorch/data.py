'''Data download function for CIFAR-10 dataset. Use to pre-download data
during devcontainer creation'''

from pathlib import Path
from torchvision import datasets

def download_cifar10_data(data_dir: str='data/pytorch/CIFAR10'):
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

if __name__ == '__main__':

    download_cifar10_data()