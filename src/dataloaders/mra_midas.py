from pathlib import Path

import torch
import cv2
from PIL import Image

import pandas as pd
import re


class MRAMIDAS(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None):
        super(MRAMIDAS, self).__init__()
        self.root = root
        self.transform = transform
        self.images = Path(self.root) / "images"
        self.metadata_path = Path(self.root) / "release_midas.xlsx"

        df = pd.read_excel(self.metadata_path)
        df["midas_path"] = df["midas_path"].fillna("No finding").astype(str)
        df['midas_file_name'] = df['midas_file_name'].str.replace('.jpeg', '.jpg', regex=False)
        df["midas_age"] = df["midas_age"].fillna("Unknown").astype(str)
        df["midas_gender"] = df["midas_gender"].fillna("Unknown").astype(str)
        df["midas_race"] = df["midas_race"].fillna("Unknown").astype(str)
        df["midas_location"] = df["midas_location"].fillna("Unknown").astype(str)
        df["midas_melanoma"] = df["midas_melanoma"].fillna("Unknown").astype(str)
        idxs_to_drop = [
            283, 313, 464, 604, 651, 661, 674, 686, 796, 797, 811, 834, 889, 904, 906, 907,
            922, 992, 1136, 1218, 1219, 1220, 1475, 1476, 1970, 2268, 2373, 2420, 2422, 2423,
            2591, 2756, 2758, 3005, 3007, 3119, 3120, 3121, 3332
        ] # a little bit of magic numbers :) # TODO: automate the finding of corrupt images during initialization
        df.drop(idxs_to_drop, inplace=True)
        self.metadata = self._extract_fitzpatrick_skin_type(df)
        unique_labels = sorted(self.metadata['midas_path'].astype(str).unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {k:v for v, k in self.class_to_idx.items()}

    @classmethod
    def get_info(cls):
        from src.dataloaders.dataset_registry import INFO
        return INFO["MRA-MIDAS"]

    @staticmethod
    def _extract_fitzpatrick_skin_type(dataframe):
        def get_fitzpatrick_number(text):
            if not isinstance(text, str):
                return None
            roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'}
            match = re.match(r'^(vi|v|iv|iii|ii|i)\b', text.strip().lower())
            if match:
                return roman_map[match.group(0)]
            return None

        dataframe['fitzpatrick_skin_type'] = dataframe['midas_fitzpatrick'].apply(get_fitzpatrick_number)
        dataframe['fitzpatrick_skin_type'] = dataframe['fitzpatrick_skin_type'].fillna('Unknown')
        return dataframe

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["midas_file_name"]
        img_path = self.images / f"{img_id}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label_str = row["midas_path"]
        label = self.class_to_idx[label_str]

        label = {
            "label": torch.tensor(label, dtype=torch.long),
            "label_str": label_str,
            "fitzpatrick": row["fitzpatrick_skin_type"],
            "age": row["midas_age"],
            "sex": row["midas_gender"],
            "race": row["midas_race"],
            "location": row["midas_location"],
            "melanoma": row["midas_melanoma"],
        }

        return img, label