from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


class DSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path= "../assets/datasets/dsprites.npz"):
        data = np.load(data_path, allow_pickle=True, encoding='latin1')

        self.images = data['imgs']
        self.labels =  data['latents_values'][:,1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image from numpy array to PyTorch tensor and add channel dimension
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx]-1, dtype=torch.int64)
        return image, label

def get_dsprites(
        path: str = "../assets/datasets/dsprites.npz", 
        batch_size: int = 32, 
        shuffle: bool = True, 
        num_workers: int = 4,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Utility function to create a DataLoader for the DSprites dataset.

    Args:
        path (str): Path to the DSprites .npz file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch. Default is True.
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.
    Returns:
        DataLoader: A PyTorch DataLoader for the DSprites dataset.
    """
    
    ds = DSpritesDataset(path)

    
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ds, test_size=test_size, random_state=seed, shuffle=shuffle)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader