from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


class Fitzpatrick(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Any = None):
        super(Fitzpatrick, self).__init__()
        self.root = root
        self.transform = transform

        self.images_dir = Path(root) / "images"
        self.labels_file = Path(root) / "labels.csv"

        df = pd.read_csv(self.labels_file)
        df["image_path"] = df["md5hash"].apply(lambda x: self.images_dir / f"{x}.jpg")
        self.labels = df[df["image_path"].apply(lambda x: x.exists())].reset_index(
            drop=True
        )

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

    @classmethod
    def get_info(cls):
        from src.dataloaders.dataset_registry import INFO
        return INFO["Fitzpatrick17k"]

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
        img_hash = row["md5hash"]
        img_path = self.images_dir / f"{img_hash}.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)


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
            "fp_scale": row["fitzpatrick_scale"],
            "fp_centaur": row.get("fitzpatrick_centaur"),
            "img_path": str(img_path)
        }
        return img, label
