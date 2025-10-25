import random
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import pandas as pd
import torch
from PIL import Image
import re
from sklearn.model_selection import train_test_split


def is_image_valid(path):
    """Checks if an image file is not corrupted."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

class MRAMIDAS(torch.utils.data.Dataset):
    def __init__(self, root: str):
        super(MRAMIDAS, self).__init__()
        self.root = root
        self.images_dir = Path(self.root) / "images"
        self.metadata_path = Path(self.root) / "release_midas.xlsx"

        df = pd.read_excel(self.metadata_path)

        df["label"] = df["midas_path"].fillna("No finding").astype(str)
        df['midas_file_name'] = df['midas_file_name'].str.replace('.jpeg', '.jpg', regex=False)
        df["age"] = df["midas_age"].fillna("Unknown").astype(str)
        df["sex"] = df["midas_gender"].fillna("Unknown").astype(str)
        df["midas_race"] = df["midas_race"].fillna("Unknown").astype(str)
        df["location"] = df["midas_location"].fillna("Unknown").astype(str)
        df["midas_melanoma"] = df["midas_melanoma"].fillna("Unknown").astype(str)
        
        print("Verifying dataset images (this may take a moment)...")
        
        df['image_path'] = df['midas_file_name'].apply(lambda x: self.images_dir / x)
        df = df[df['image_path'].apply(lambda x: x.exists())].reset_index(drop=True)
        print(f"Found {len(df)} existing image files.")
        tqdm.pandas(desc="Checking image integrity") 
        
        is_valid = df['image_path'].progress_apply(is_image_valid)
        
        corrupted_count = len(is_valid) - is_valid.sum()
        if corrupted_count > 0:
            print(f"Found and removed {corrupted_count} corrupted images.")
        
        self.metadata = df[is_valid].reset_index(drop=True)
        
        print(f"Dataset ready with {len(self.metadata)} valid images.")

        self.metadata = self._extract_fitzpatrick_skin_type(self.metadata)
        self.metadata.rename(columns={"fitzpatrick_skin_type": "fitzpatrick_type"}, inplace=True)

        from src.backbones import generate_prompts_for_clip
        self.metadata["clip_label"] = self.metadata.apply(generate_prompts_for_clip, axis=1)

        unique_labels = sorted(self.metadata['label'].astype(str).unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    @staticmethod
    def _extract_fitzpatrick_skin_type(dataframe):
        def get_fitzpatrick_number(text):
            if not isinstance(text, str):
                return None
            roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'}
            match = re.match(r'^(vi|v|iv|iii|ii|i)\b', text.strip().lower())
            if match:
                return int(roman_map[match.group(0)])
            return None

        dataframe['fitzpatrick_skin_type'] = dataframe['midas_fitzpatrick'].apply(get_fitzpatrick_number)
        return dataframe

    def label_to_idx(self, label_value: str, label_type: str = "label") -> int:
        """Converts a label string to its corresponding index."""
        return self.class_to_idx[label_value]

    def idx_to_label(self, idx: int, label_type: str = "label") -> str:
        """Converts a label index to its corresponding string."""
        return self.idx_to_class[idx]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")

        disease_label = row["label"]
        fitzpatrick_type = row["fitzpatrick_type"]


        label_idx = self.label_to_idx(label_value=disease_label)

        label = {
            "label": torch.tensor(label_idx, dtype=torch.long),
            "label_str": disease_label,
            "clip_label": random.sample(row["clip_label"], k=1)[0] if isinstance(row["clip_label"], list) else row["clip_label"],
            "fp_scale": fitzpatrick_type,
            "age": row["age"],
            "sex": row["sex"],
            "race": row["midas_race"],
            "location": row["location"],
            "melanoma": row["midas_melanoma"],
            "img_path": str(img_path)
        }

        return img, label
    
def get_mra_midas(
        root: str,
        collate_fn=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Instantiates the MRAMIDAS dataset and returns train and validation DataLoaders.
    """
    ds = MRAMIDAS(root=root)

    train_ds, val_ds = train_test_split(
        ds,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader