import torch
import pandas as pd
import os
import random
from pathlib import Path
from PIL import Image
from typing import Tuple




class HAM10K(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(HAM10K, self).__init__()
        self.root = root
        self.transform = transform
        self.images = Path(self.root) / 'images'
        self.metadata = pd.read_csv(os.path.join(root, "HAM10000_metadata"))
        self.metadata.rename(columns={"localization": "location"}, inplace=True)

        from src.backbones import generate_prompts_for_clip
        self.metadata["clip_label"] = self.metadata.apply(generate_prompts_for_clip, axis=1)


        self.class_to_idx = {
            "bkl": 0,
            "nv": 1,
            "df": 2,
            "mel": 3,
            "vasc": 4,
            "bcc": 5,
            "akiec": 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset at the given index.
        """
        row = self.metadata.iloc[idx]
        img_id = row["image_id"]
        img_path = self.images / f"{img_id}.jpg"
        
        img = Image.open(img_path).convert("RGB")

        label_str = row["dx"]
        label_idx = self.class_to_idx[label_str]
        
        label = {
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_str": label_str,
            "clip_label": random.sample(row["clip_label"], k=1)[0] if isinstance(row["clip_label"], list) else row["clip_label"],
            "label_type": row["dx_type"],
            "age": row["age"],
            "sex": row["sex"],
            "location": row["location"],
            "img_path": str(img_path)
        }

        return img, label
    
def get_ham10000(
        root: str, 
        collate_fn=None, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        num_workers: int = 4,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds = HAM10K(root=root)
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(ds, test_size=test_size, random_state=seed, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader