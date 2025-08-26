from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset



class FitzpatrickSKINCON(Dataset):
    def __init__(self, root: str, transform: Any = None):
        super(FitzpatrickSKINCON, self).__init__()
        self.root = root
        self.transform = transform

        self.images_dir = Path(root) / "images"
        labels_df = pd.read_csv(Path(root) / "labels.csv")
        skincon_df = pd.read_csv(Path(root) / "skincon-annotations.csv", index_col=0)

        excluded_cols = ['ImageID', '<anonymous>', 'Do not consider this image']
        self.concept_columns = sorted([
            col for col in skincon_df.columns if col not in excluded_cols
        ])
        skincon_df[self.concept_columns] = skincon_df[self.concept_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Keep only the rows where at least one concept is present (sum > 0)
        skincon_df = skincon_df[skincon_df[self.concept_columns].sum(axis=1) > 0]
        skincon_prepared = skincon_df[self.concept_columns + ['ImageID']].set_index('ImageID')

        labels_df["ImageID"] = labels_df["md5hash"].astype(str) + ".jpg"
        labels_df["image_path"] = labels_df["ImageID"].apply(lambda x: self.images_dir / x)
        master_df = labels_df[labels_df["image_path"].apply(lambda x: x.exists())]

        self.metadata = master_df.set_index('ImageID').join(skincon_prepared, how='inner').reset_index()

        self.metadata[self.concept_columns] = self.metadata[self.concept_columns].fillna(0)

        self.label_to_idx_maps = {}
        self.idx_to_label_maps = {}
        for label_type in ["label", "nine_partition_label", "three_partition_label"]:
            if label_type in self.metadata.columns:
                unique_values = sorted(self.metadata[label_type].dropna().unique())
                self.label_to_idx_maps[label_type] = {val: idx for idx, val in enumerate(unique_values)}
                self.idx_to_label_maps[label_type] = {idx: val for val, idx in self.label_to_idx_maps[label_type].items()}

    def get_concepts(self, concepts_tensor):
        """Translates a multi-hot encoded tensor back to a list of concept names."""
        present_indices = torch.where(concepts_tensor == 1)[1]
        present_concepts = [self.concept_columns[i] for i in present_indices]

        return present_concepts

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
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row["image_path"]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        concepts_vector = row[self.concept_columns].values.astype(float)
        concepts_tensor = torch.tensor(concepts_vector, dtype=torch.float32)

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
            "concepts": concepts_tensor,
        }
        return img, label