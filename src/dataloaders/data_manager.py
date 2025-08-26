from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.dataloaders import Fitzpatrick, HAM10K, SCIN, MRAMIDAS, FitzpatrickSKINCON
from src.dataloaders.index_dataset import IndexedDataset
from src.dataloaders.isic2020 import ISIC2020
from src.dataloaders.ddi import DDI


class ImageDataManager:
    """
    Data manager class that yields dataloaders for self-supervised pretraining.

    Args:
        data_root (str): Root directory of the dataset. Defaults to 'data'.
        initialize (str | list): Which datasets to initialize. Defaults to 'all'.
        seed (int): Random seed. Defaults to 42.
        transform (callable, optional): Transformation function that takes an image and returns the PyTorch compatible
                                        transformed image.
    """
    available_datasets: dict = {
        "scin": SCIN,
        "fitzpatrick17k": Fitzpatrick,
        "skincon_fitzpatrick17k": FitzpatrickSKINCON,
        "ham10000": HAM10K,
        "mra-midas": MRAMIDAS,
        "isic2020": ISIC2020,
        "ddi": DDI,
    }

    def __init__(
            self,
            data_root: str = "data",
            initialize: str | list = "all",
            seed: int = 42,
            transform: Any = None,
    ):
        self._check_if_supported(initialize)
        self.data_root = Path(data_root)
        self.datasets = {}
        self.seed = seed
        self.transform = transform
        self.init_dataset(initialize)

    def init_dataset(self, dataset: str | list):
        """Initializes the datasets."""
        if isinstance(dataset, str) and dataset != "all":
            ds = self._get_dataset(dataset, transform=self.transform)
            self.datasets[dataset] = ds
        elif isinstance(dataset, list) or dataset == "all":
            if dataset == "all":
                datasets = self.available_datasets.keys()
            else:
                datasets = dataset
            for d in datasets:
                ds = self._get_dataset(d, transform=self.transform)
                self.datasets[d] = ds

    def _get_dataset(self, dataset: str, transform: Any = None) -> Dataset:
        """Fetches and prepares a single dataset."""
        if dataset == "skincon_fitzpatrick17k":
            dataset_path = self.data_root / "fitzpatrick17k"
        elif dataset == "isic2020":
            dataset_path = self.data_root / "ISIC_2020_Training_JPEG"
        else:
            dataset_path = self.data_root / dataset.lower()
        return self.available_datasets[dataset.lower()](root=dataset_path, transform=transform)


    def get_dataset_infos(self, dataset: str = None) -> dict:
        """Returns infos about the available datasets."""
        if dataset:
            return self.datasets[dataset].get_info()
        elif dataset is None:
            info_dict = {}
            for dataset in self.datasets.keys():
                 info_dict[dataset] = self.datasets[dataset].get_info()
            return info_dict
        else:
            raise ValueError("Dataset not available.")

    def get_indexed_dataset(
            self,
            dataset: str | list = None,
    ):
        """
        Wraps a dataset with an IndexedDataset that yields (index, *data) tuples.

        Args:
            dataset (str | list): Which datasets to yield. Defaults to 'all'.

        Returns:
            IndexedDataset: IndexedDataset that yields (index, *data) tuples.
        """

        if isinstance(dataset, str) and dataset.lower() in self.datasets.keys():
            dset = self.datasets[dataset]

        elif isinstance(dataset, list):
            print("Generating combined pretraining dataset...")
            dsets = []
            for d in dataset:
                assert d.lower() in self.datasets.keys(), f"Dataset {d} is not available"
                dsets.append(self.datasets[d])
            dset = ConcatDataset(dsets)
        else:
            raise ValueError("Dataset not available.")

        return IndexedDataset(dset)


    def get_dataloaders(
            self,
            dataset: str | list,
            batch_size: int = 64,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False,
            test_size: float = 0.05
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Yields the train and test dataloaders.

        Args:
            dataset (str | list): Which datasets to yield. Defaults to 'all'.
            batch_size (int): Batch size for training. Defaults to 64.
            shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            num_workers (int): Number of workers. Defaults to 0.
            pin_memory (bool): Whether to pin memory. Defaults to False.
            test_size (float): Ratio of dataset to test size. Defaults to 0.05.

        Returns:
            Tuple[DataLoader, DataLoader]: Tuple of train and test dataloaders.
        """
        if isinstance(dataset, str) and dataset.lower() in self.datasets.keys():
            dset = self.datasets[dataset]

            indices = np.arange(len(dset))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            train_size = int((1 - test_size) * len(indices))
            train_idx, test_idx = indices[:train_size], indices[train_size:]

            train_dset, test_dset = (
                torch.utils.data.Subset(dset, train_idx),
                torch.utils.data.Subset(dset, test_idx),
            )

            train_loader = torch.utils.data.DataLoader(
                train_dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
            )
            test_loader = torch.utils.data.DataLoader(
                test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            return train_loader, test_loader

        elif isinstance(dataset, list):
            print("Generating combined pretraining dataset...")
            dsets = []
            for d in dataset:
                assert d.lower() in self.datasets.keys(), f"Dataset {d} is not available"
                dsets.append(self.datasets[d])
            concatenated_dsets = ConcatDataset(dsets)
            indices = np.arange(len(concatenated_dsets))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            train_size = int((1 - test_size) * len(indices))
            train_idx, test_idx = indices[:train_size], indices[train_size:]

            train_dset, test_dset = (
                torch.utils.data.Subset(concatenated_dsets, train_idx),
                torch.utils.data.Subset(concatenated_dsets, test_idx),
            )

            train_loader = torch.utils.data.DataLoader(
                train_dset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=pretrain_collate_fn
            )
            test_loader = torch.utils.data.DataLoader(
                test_dset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, collate_fn=pretrain_collate_fn
            )
            return train_loader, test_loader
        else:
            raise ValueError("Dataset not available.")

    def _check_if_supported(self, initialize):
        """Checks if dataset is in supported datasets."""
        if isinstance(initialize, str):
            assert (initialize.lower() in ImageDataManager.available_datasets.keys()
                    or initialize.lower() == "all"), \
                f"Initialization must be 'all' or one of {self.available_datasets.keys()}."
        elif isinstance(initialize, list):
            for initialize_item in initialize:
                assert (initialize_item.lower() in ImageDataManager.available_datasets.keys()), \
                    f"Initialization must be 'all' or one of {self.available_datasets.keys()}."
        else:
            raise ValueError(f"Initialization must be 'all' or one of {self.available_datasets.keys()}.")


def pretrain_collate_fn(batch):
    """
    Custom collate function that correctly stacks tensors from metadata dictionaries.
    'batch' is a list of tuples, where each tuple is (image, metadata_dict).
    """
    images = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]
    images_batch = torch.stack(images, 0)

    metadata_batch = {}

    # Get all unique keys from all metadata dictionaries in the batch
    all_keys = set().union(*[d.keys() for d in metadata_list])

    for key in all_keys:
        values = [d.get(key) for d in metadata_list if d.get(key) is not None]

        if not values:
            continue

        if isinstance(values[0], torch.Tensor):
            metadata_batch[key] = torch.stack(values, 0)
        else:
            metadata_batch[key] = values

    return images_batch, metadata_batch


def indexed_collate_fn(batch):
    """
    Custom collate function to handle dictionaries with varying keys.
    'batch' is a list of tuples, where each tuple is (image, metadata_dict).
    """
    indices = [item[0] for item in batch]
    images = [item[1] for item in batch]
    metadata_list = [item[2] for item in batch]

    indices_batch = torch.stack(indices, dim=0)
    images_batch = torch.stack(images, 0)

    metadata_batch = {}

    all_keys = set().union(*[d.keys() for d in metadata_list])

    for key in all_keys:
        metadata_batch[key] = [d.get(key, None) for d in metadata_list]

    return indices_batch, images_batch, metadata_batch