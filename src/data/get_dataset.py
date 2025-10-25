import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Literal

from src.data.dataset import get_combined_dataset, DermDataset

def get_dataloaders(
    datasets: List,
    data_root: str,
    transform, 
    batch_size: int = 32,
    num_workers: int = 8,
    test_size: float = 0.2,
    seed: int = 42,
    use_weighted_sampling: bool = True,
    labelkey="label"
):
    """
    Creates training and validation dataloaders from a unified master CSV file.

    It still uses weighted sampling for balanced training.
    """

    df = get_combined_dataset(datasets, data_root=data_root)
    
    print(f"Performing naive random split with test_size={test_size}...")
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    print(f"Split complete: {len(train_df)} training samples, {len(val_df)} validation samples.")

    train_dataset = DermDataset(dataframe=train_df, transform=transform, labelkey=labelkey)
    val_dataset = DermDataset(dataframe=val_df, transform=transform, labelkey=labelkey)

    sampler = None
    shuffle = not use_weighted_sampling

    if use_weighted_sampling:
        print("Calculating weights for balanced sampling...")
        class_counts = train_df['source_dataset'].value_counts()
        sample_weights = [1.0 / class_counts[source] for source in train_df['source_dataset']]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        print("WeightedRandomSampler created.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=torch.utils.data.default_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=torch.utils.data.default_collate,
        pin_memory=True
    )
    
    return train_loader, val_loader