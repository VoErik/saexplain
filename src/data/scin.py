import ast
import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image

from sklearn.model_selection import train_test_split
from typing import Tuple


GENERAL_TEMPLATES = [
    "a photo of {label}.",
    "an image of {label}.",
    "a dermatological image showing {label}.",
    "a clinical presentation of {label}.",
    "a skin condition known as {label}.",
    "an example of what {label} looks like.",
    "skin with a condition identified as {label}."
]

SKIN_TYPE_TEMPLATES = [
    "a photo of {label} on Fitzpatrick skin type {type}.",
    "a clinical image of {label} on skin type {type}.",
    "{label} as it appears on type {type} skin.",
    "a case of {label} affecting a person with Fitzpatrick type {type} skin."
]


class SCIN(torch.utils.data.Dataset):
    def __init__(self, root: str):
        super(SCIN, self).__init__()
        self.root = root
        self.images_dir = Path(self.root)
        self.labels_file = Path(self.root) / "scin_labels.csv"
        self.cases_file = Path(self.root) / "scin_cases.csv"

        cases_df = pd.read_csv(self.cases_file)
        cases_df['case_id'] = cases_df['case_id'].astype(str)
        labels_df = pd.read_csv(self.labels_file)
        labels_df['case_id'] = labels_df['case_id'].astype(str)
        merged_df = pd.merge(cases_df, labels_df, on='case_id')
        
        image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
        id_columns = [col for col in merged_df.columns if col not in image_path_columns]
        df_melted = pd.melt(merged_df, id_vars=id_columns, value_vars=image_path_columns,
                            var_name="original_cols", value_name="image_path")
        df_final = df_melted.dropna(subset=['image_path']).drop(columns='original_cols')
        
        df_final["image_path"] = df_final["image_path"].str.replace(
            'dataset', str(self.root), n=1
        )

        def get_first_condition(list_string):
            try:
                conditions = ast.literal_eval(list_string)
                return conditions[0] if conditions else "No finding"
            except (ValueError, SyntaxError):
                return "No finding"
                
        df_final['label'] = df_final['dermatologist_skin_condition_on_label_name'].apply(get_first_condition)
        df_final['fitzpatrick_type'] = pd.to_numeric(df_final['dermatologist_fitzpatrick_skin_type_label_1'], errors='coerce')

        df_final["sex"] = df_final["sex_at_birth"].fillna("Unknown")
        df_final["age"] = df_final["age_group"].fillna("Unknown")

        print("Verifying SCIN dataset images...")
        self.df = df_final[df_final['image_path'].apply(lambda x: Path(x).exists())].reset_index(drop=True)
        print(f"Found {len(self.df)} valid images out of {len(df_final)} total.")

        from src.backbones import generate_prompts_for_clip
        self.df["clip_label"] = self.df.apply(generate_prompts_for_clip, axis=1)

        unique_labels = sorted(self.df['label'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def label_to_idx(self, label_value: str, label_type: str = "label") -> int:
        return self.class_to_idx[label_value]

    def idx_to_label(self, idx: int, label_type: str = "label") -> str:
        return self.idx_to_class[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
            "age": row["age_group"],
            "sex": row["sex_at_birth"],
            "race": row.get("combined_race", None),
            "location": None,
            "img_path": str(img_path)
        }
        return img, label



def get_scin(
        root: str,
        collate_fn=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Instantiates the SCIN dataset and returns train and validation DataLoaders.
    """
    ds = SCIN(root=root)
    
    # Stratified split to handle class imbalance if possible
    try:
        train_ds, val_ds = train_test_split(
            ds,
            test_size=test_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=ds.df['label'].tolist() # Stratify based on the main disease label
        )
    except ValueError:
        print("Warning: Could not stratify split. Using random split instead.")
        train_ds, val_ds = train_test_split(ds, test_size=test_size, random_state=seed, shuffle=shuffle)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, val_loader