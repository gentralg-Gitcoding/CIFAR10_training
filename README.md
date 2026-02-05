# PyTorch: CIFAR-10 Demonstration

[![Publish to PyPI](https://github.com/gperdrizet/CIFAR10/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/gperdrizet/CIFAR10/actions/workflows/publish-to-pypi.yml)

[![Deploy Documentation](https://github.com/gperdrizet/CIFAR10/actions/workflows/docs.yml/badge.svg)](https://github.com/gperdrizet/CIFAR10/actions/workflows/docs.yml)

A progressive deep learning tutorial for image classification on the CIFAR-10 dataset using PyTorch. This project demonstrates the evolution from basic deep neural networks to optimized convolutional neural networks with data augmentation. It also provides a set of utility functions as a PyPI package for use in other projects.

[View on PyPI](https://pypi.org/project/cifar10_tools) | [Documentation](https://gperdrizet.github.io/CIFAR10/)

## Installation

Install the helper tools package locally in editable mode to use in this repository:

```bash
pip install -e .
```

Or install from PyPI to use in other projects:

```bash
pip install cifar10_tools
```

## Project overview

This repository contains a series of Jupyter notebooks that progressively build more sophisticated neural network architectures for the CIFAR-10 image classification task. Each notebook builds upon concepts from the previous one, demonstrating key deep learning techniques.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01-DNN.ipynb](notebooks/01-DNN.ipynb) | **Deep Neural Network** - Baseline fully-connected DNN classifier using `nn.Sequential`. Establishes a performance baseline with a simple architecture. |
| [02-CNN.ipynb](notebooks/02-CNN.ipynb) | **Convolutional Neural Network** - Introduction to CNNs with convolutional and pooling layers using `nn.Sequential`. Demonstrates the advantage of CNNs over DNNs for image tasks. |
| [03-RGB-CNN.ipynb](notebooks/03-RGB-CNN.ipynb) | **RGB CNN** - CNN classifier that utilizes full RGB color information instead of grayscale, improving feature extraction from color images. |
| [04-optimized-CNN.ipynb](notebooks/04-optimized-CNN.ipynb) | **Hyperparameter Optimization** - Uses Optuna for automated hyperparameter tuning to find optimal network architecture and training parameters. |
| [05-augmented-CNN.ipynb](notebooks/05-augmented-CNN.ipynb) | **Data Augmentation** - Trains the optimized CNN architecture with image augmentation techniques for improved generalization and robustness. |

## Requirements

- Python >=3.10, <3.13
- PyTorch >=2.0
- torchvision >=0.15
- numpy >=1.24

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.