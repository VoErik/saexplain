import torch
from pathlib import Path
import pandas as pd
import os
import cv2
from PIL import Image


class ISIC2020(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(ISIC2020, self).__init__()
        self.root = root
        self.transform = transform
        self.images = Path(self.root) / "train"
        self.metadata = pd.read_csv(os.path.join(root, "ISIC_2020_Training_GroundTruth_v2.csv"))
        self.class_to_idx = {
            "benign": 0,
            "malignant": 1,
        }
        self.idx_to_class = {k: v for v, k in self.class_to_idx.items()}


    @classmethod
    def get_info(cls):
        from src.dataloaders.dataset_registry import INFO
        return INFO["ISIC2020"]

    def __len__(self):
        return len(self.metadata["lesion_id"])


    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["image_name"]
        img_path = self.images / f"{img_id}.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label_str = row["benign_malignant"]
        label = self.class_to_idx[label_str]

        label = {
            "label": torch.tensor(label, dtype=torch.long),
            "label_str": label_str,
            "age": row["age_approx"],
            "sex": row["sex"],
            "location": row["anatom_site_general_challenge"],
        }

        return img, label