import math
import os
from typing import Any

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src.dataloaders.transforms import DATASET_STATISTICS, denormalize


def visualize_batch(
    images: torch.Tensor,
    labels: list,
    label_map: dict = None,
    title: str = "Image Batch",
    nrow: int = 8,
    save_path: str = None,
    visualize: bool = True,
):
    """
    Visualize a batch of images with their labels.

    Args:
        images (Tensor): A batch of images, shape (B, C, H, W) or (B, H, W, C).
        labels (list): A list of labels corresponding to the images.
        label_map (dict, optional): Maps label indices to names, e.g., {0: 'cat', 1: 'dog'}.
        title (str): Title of the plot.
        nrow (int): Number of images per row.
    """
    # Convert to channel-first if needed
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    # Normalize if needed (assumes [0, 1] range)
    images = torch.clamp(images, 0, 1)

    B, C, H, W = images.shape
    ncols = nrow
    nrows = math.ceil(B / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten() if B > 1 else [axes]

    for idx in range(nrows * ncols):
        ax = axes[idx]
        ax.axis("off")

        if idx < B:
            img = images[idx].permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)

            label = labels[idx]
            label_str = label_map[label] if label_map else str(label)
            ax.set_title(label_str, fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300)
    if visualize:
        plt.show()
    plt.close()


def visualize_with_skincon(loader, name: str, dset: Any):
    """Prints a denormalized batch of images and their labels."""
    mean = DATASET_STATISTICS[name]["mean"]
    std = DATASET_STATISTICS[name]["std"]

    for img, label in loader:
        print("Original shape:", img.shape)
        img = denormalize(img, mean, std)
        print("Shape after denormalization:", img.shape)

        grid = make_grid(img)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()

        print("Labels:", label["label"])
        print("Label strings:", label["label_str"])
        print("Concepts:", dset.get_concepts(label["concepts"]))
        break