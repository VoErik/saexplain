from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import random
import kornia.filters as kf

from src.dataloaders.dataset_registry import INFO
import torch


DATASET_STATISTICS = {
    "SCIN": {
        "mean": INFO["SCIN"]["mean"],
        "std": INFO["SCIN"]["std"],
    },
    "Fitzpatrick17k": {
        "mean": INFO["Fitzpatrick17k"]["mean"],
        "std": INFO["Fitzpatrick17k"]["std"],
    },
    "MRA-MIDAS": {
        "mean": INFO["MRA-MIDAS"]["mean"],
        "std": INFO["MRA-MIDAS"]["std"],
    },
    "HAM10000": {
        "mean": INFO["HAM10000"]["mean"],
        "std": INFO["HAM10000"]["std"],
    },
    "ClinicalPhotos": {
        "mean": INFO["ClinicalPhotos"]["mean"],
        "std": INFO["ClinicalPhotos"]["std"],
    },
    "DDI": {
        "mean": INFO["DDI"]["mean"],
        "std": INFO["DDI"]["std"],
    }
}

def get_transforms(size: int = 224):
    #mean = DATASET_STATISTICS[dataset]["mean"]
    #std = DATASET_STATISTICS[dataset]["std"]
    return derma_transform(size=size)

def target_transform(size: int):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    return transform

def imagenet_transform(size):
    """
    Creates a transforms.Compose object that applies the standard ImageNet transform.

    This includes resizing, center cropping, converting to a tensor, and normalizing.

    :return: A torchvision.transforms.Compose object.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return transform

class MotionBlur:
    def __init__(self, kernel_size: int, angle: tuple[float, float], direction: float = 0.5):
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_batch = img.unsqueeze(0)
        angle = random.uniform(self.angle[0], self.angle[1])
        blurred_batch = kf.motion_blur(img_batch, self.kernel_size, angle, self.direction)
        return blurred_batch.squeeze(0)

def derma_transform(size: int = 224):
    """
    Some common augmentations applied to dermoscopic images. Taken from DiSalvo (2024): MedMNIST-C.
    """
    corruption_list = [
        transforms.ColorJitter(brightness=0.4),
        transforms.ColorJitter(contrast=0.4),
        transforms.ColorJitter(saturation=0.4),
        transforms.ColorJitter(hue=0.1),
        MotionBlur(kernel_size=25, angle=(-45, 45)),
    ]

    corruption_transform = transforms.RandomApply(
        [transforms.RandomChoice(corruption_list)],
        p=0.65
    )

    final_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        corruption_transform,
    ])
    return final_transform

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor image with mean and standard deviation.

    Args:
        tensor (torch.Tensor): The normalized image tensor (C, H, W).
        mean (list or tuple): The mean used for normalization.
        std (list or tuple): The standard deviation used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    denormalized_tensor = tensor * std + mean
    return torch.clamp(denormalized_tensor, 0, 1)


def print_batch(loader, name: str):
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
        break