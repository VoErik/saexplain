from pathlib import Path

import pandas as pd
import torch
import os

from PIL import Image
import cv2



class HAM10K(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(HAM10K, self).__init__()
        self.root = root
        self.transform = transform
        self.images = Path(self.root) / 'images'
        self.metadata = pd.read_csv(os.path.join(root, "HAM10000_metadata"))
        self.class_to_idx = {
            "bkl": 0,
            "nv": 1,
            "df": 2,
            "mel": 3,
            "vasc": 4,
            "bcc": 5,
            "akiec": 6
        }
        self.idx_to_class = {k: v for v, k in self.class_to_idx.items()}

    @classmethod
    def get_info(cls):
        from src.dataloaders.dataset_registry import INFO
        return INFO["HAM10000"]

    def __len__(self):
        return len(self.metadata["lesion_id"])

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["image_id"]
        img_path = self.images / f"{img_id}.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label_str = row["dx"]
        label = self.class_to_idx[label_str]

        label = {
            "label": torch.tensor(label,dtype=torch.long),
            "label_str": label_str,
            "label_type": row["dx_type"],
            "age": row["age"],
            "sex": row["sex"],
            "location": row["localization"],
            "img_path": str(img_path)
        }

        return img, label
