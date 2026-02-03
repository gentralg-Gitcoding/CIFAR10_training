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


def plot_learning_curves(
    history: dict[str, list[float]],
    figsize: tuple[float, float] = (10, 4)
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 
                 'train_accuracy', and 'val_accuracy' lists.
        figsize: Figure size (width, height).
        
    Returns:
        Tuple of (figure, axes array).
    '''
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].set_title('Loss')
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (cross-entropy)')
    axes[0].legend(loc='best')

    axes[1].set_title('Accuracy')
    axes[1].plot(history['train_accuracy'], label='Train')
    axes[1].plot(history['val_accuracy'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='best')

    plt.tight_layout()
    
    return fig, axes


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    figsize: tuple[float, float] = (8, 8),
    cmap: str = 'Blues'
) -> tuple[plt.Figure, plt.Axes]:
    '''Plot a confusion matrix heatmap.
    
    Args:
        true_labels: Array of true class labels.
        predictions: Array of predicted class labels.
        class_names: List of class names for labeling.
        figsize: Figure size (width, height).
        cmap: Colormap for the heatmap.
        
    Returns:
        Tuple of (figure, axes).
    '''
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, predictions)
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title('Confusion matrix')
    im = ax.imshow(cm, cmap=cmap)

    # Add labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.tight_layout()
    
    return fig, ax


def plot_class_probability_distributions(
    all_probs: np.ndarray,
    class_names: list[str],
    nrows: int = 2,
    ncols: int = 5,
    figsize: tuple[float, float] = (12, 4),
    bins: int = 50,
    color: str = 'black'
) -> tuple[plt.Figure, np.ndarray]:
    '''Plot predicted probability distributions for each class.
    
    Args:
        all_probs: Array of shape (n_samples, n_classes) with predicted probabilities.
        class_names: List of class names for labeling.
        nrows: Number of rows in the subplot grid.
        ncols: Number of columns in the subplot grid.
        figsize: Figure size (width, height).
        bins: Number of histogram bins.
        color: Histogram bar color.
        
    Returns:
        Tuple of (figure, axes array).
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    fig.suptitle('Predicted probability distributions by class', fontsize=14, y=1.02)
    fig.supxlabel('Predicted probability', fontsize=12)
    fig.supylabel('Count', fontsize=12)

    axes = axes.flatten()

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Get probabilities for this class across all samples
        class_probs = all_probs[:, i]
        
        # Plot histogram
        ax.hist(class_probs, bins=bins, color=color)
        ax.set_title(class_name)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    
    return fig, axes