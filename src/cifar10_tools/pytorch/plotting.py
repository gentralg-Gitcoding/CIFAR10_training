'''Plotting functions for CIFAR-10 models.'''

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


def plot_sample_images(
    dataset: Dataset,
    class_names: list[str],
    nrows: int = 2,
    ncols: int = 5,
    figsize: tuple[float, float] | None = None,
    cmap: str = 'gray'
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot sample images from a dataset.
    
    Args:
        dataset: PyTorch dataset containing (image, label) tuples.
        class_names: List of class names for labeling.
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        figsize: Figure size (width, height). Defaults to (ncols*1.5, nrows*1.5).
        cmap: Colormap for displaying images.
        
    Returns:
        Tuple of (figure, axes array).
    '''
    if figsize is None:
        figsize = (ncols * 1.5, nrows * 1.5)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Get image and label from dataset
        img, label = dataset[i]
        
        # Unnormalize and squeeze for plotting
        img = img * 0.5 + 0.5
        img = img.numpy().squeeze()
        ax.set_title(class_names[label])
        ax.imshow(img, cmap=cmap)
        ax.axis('off')

    plt.tight_layout()
    
    return fig, axes
