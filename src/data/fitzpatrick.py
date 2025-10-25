import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image


class Fitzpatrick(torch.utils.data.Dataset):
    def __init__(self, root: str):
        super(Fitzpatrick, self).__init__()
        self.root = root

        self.images_dir = Path(root) / "images"
        self.labels_file = Path(root) / "labels.csv"

        df = pd.read_csv(self.labels_file)
        df["image_path"] = df["md5hash"].apply(lambda x: self.images_dir / f"{x}.jpg")
        df.rename(columns={"fitzpatrick_scale": "fitzpatrick_type"}, inplace=True)
        self.labels = df[df["image_path"].apply(lambda x: x.exists())].reset_index(
            drop=True
        )
        
        from src.backbones import generate_prompts_for_clip
        self.labels["clip_label"] = self.labels.apply(generate_prompts_for_clip, axis=1)

        self.label_to_idx_maps = {
            "label": {
                label: idx for idx, label in enumerate(self.labels["label"].unique())
            },
            "nine_partition_label": {
                label: idx
                for idx, label in enumerate(
                    self.labels["nine_partition_label"].unique()
                )
            },
            "three_partition_label": {
                label: idx
                for idx, label in enumerate(
                    self.labels["three_partition_label"].unique()
                )
            },
        }

        self.idx_to_label_maps = {
            "label": {
                idx: label for label, idx in self.label_to_idx_maps["label"].items()
            },
            "nine_partition_label": {
                idx: label
                for label, idx in self.label_to_idx_maps["nine_partition_label"].items()
            },
            "three_partition_label": {
                idx: label
                for label, idx in self.label_to_idx_maps[
                    "three_partition_label"
                ].items()
            },
        }

    def label_to_idx(self, label_value: str, label_type: str = "label") -> int:
        """Converts a label string to its corresponding index."""
        return self.label_to_idx_maps[label_type][label_value]

    def idx_to_label(self, idx: int, label_type: str = "label") -> str:
        """Converts a label index to its corresponding string."""
        return self.idx_to_label_maps[label_type][idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")

        

        label_idx = self.label_to_idx(label_type="label", label_value=row["label"])
        nine_partition_idx = self.label_to_idx(label_type="nine_partition_label", label_value=row["nine_partition_label"])
        three_partition_idx = self.label_to_idx(label_type="three_partition_label", label_value=row["three_partition_label"])


        label = {
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_str": row["label"],
            "nine_partition_label": torch.tensor(nine_partition_idx, dtype=torch.long),
            "nine_partition_label_str": row["nine_partition_label"],
            "three_partition_label": torch.tensor(three_partition_idx, dtype=torch.long),
            "three_partition_label_str": row["three_partition_label"],
            "fp_scale": row["fitzpatrick_type"],
            "fp_centaur": row.get("fitzpatrick_centaur"),
            "img_path": str(img_path),
            "clip_label": random.sample(row["clip_label"], k=1)[0] if isinstance(row["clip_label"], list) else row["clip_label"]
        }
        return img, label # pil.img, dict
    

def get_fitzpatrick(
        root: str, 
        collate_fn=None, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        num_workers: int = 4,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds = Fitzpatrick(root=root)
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