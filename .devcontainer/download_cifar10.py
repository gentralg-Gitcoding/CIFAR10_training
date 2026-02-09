'''Download CIFAR-10 dataset using torchvision.datasets, to be run in devcontainer.'''

from pathlib import Path

from torchvision import datasets


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


if __name__ == '__main__':

    download_cifar10_data()