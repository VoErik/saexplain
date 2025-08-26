from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image

from src.dataloaders.dataset_registry import INFO


class DDI(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(DDI, self).__init__()
        self.root = root
        self.transform = transform

        self.images = Path(self.root) / "images"
        self.ddi_annotation_path = Path(self.root) / "ddi_metadata.csv"
        ddi_annotation = pd.read_csv(self.ddi_annotation_path, index_col=0)

        self.skincon_annotation_path = Path(self.root) / "skincon_annotation.csv"
        skincon_annotation = pd.read_csv(self.skincon_annotation_path, index_col=0)

        excluded_cols = ['ImageID', '<anonymous>', 'Do not consider this image']
        self.concept_columns = sorted([
            col for col in skincon_annotation.columns if col not in excluded_cols
        ])
        skincon_prepared = skincon_annotation[self.concept_columns + ['ImageID']].set_index('ImageID')
        ddi_cols_to_keep = ['DDI_file', 'disease', 'skin_tone', 'malignant']
        ddi_prepared = ddi_annotation[ddi_cols_to_keep].set_index('DDI_file')
        merged_df = skincon_prepared.join(ddi_prepared, how='left')
        self.metadata = merged_df.reset_index().rename(columns={'index': 'DDI_file'})

        unique_labels = sorted(self.metadata['disease'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {k:v for v, k in self.class_to_idx.items()}

    def get_concepts(self, concepts_tensor):
        """Translates a multi-hot encoded tensor back to a list of concept names."""
        present_indices = torch.where(concepts_tensor == 1)[1]
        present_concepts = [self.concept_columns[i] for i in present_indices]

        return present_concepts

    @classmethod
    def get_info(cls):
        return INFO["DDI"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["ImageID"]
        img_path = self.images / f"{img_id}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        concepts_vector = row[self.concept_columns].values.astype(float)
        concepts_tensor = torch.tensor(concepts_vector, dtype=torch.float32)

        label_str = row["disease"]
        label = self.class_to_idx[label_str]

        label = {
            "label": torch.tensor(label, dtype=torch.long),
            "label_str": label_str,
            "fitzpatrick": row["skin_tone"],
            "malignant": row["malignant"],
            "concepts": concepts_tensor,
        }

        return img, label