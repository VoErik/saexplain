import ast
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image



class SCIN(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(SCIN, self).__init__()
        self.root = root
        self.transform = transform
        self.images = Path(self.root) / "images"
        self.labels = Path(self.root) / "scin_labels.csv"
        self.cases = Path(self.root) / "scin_cases.csv"

        cases_df = pd.read_csv(self.cases)
        cases_df['case_id'] = cases_df['case_id'].astype(str)
        labels_df = pd.read_csv(self.labels)
        labels_df['case_id'] = labels_df['case_id'].astype(str)
        merged_df = pd.merge(cases_df, labels_df, on='case_id')
        image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
        id_columns = [col for col in merged_df.columns if col not in image_path_columns]
        df_melted = pd.melt(merged_df, id_vars=id_columns, value_vars=image_path_columns,
                            var_name="original_cols", value_name="image_path"
                            )
        df_final = df_melted.dropna(subset=['image_path'])
        df_final = df_final.drop(columns='original_cols')
        df_final = df_final.sort_values('case_id').reset_index(drop=True)
        df_final["image_path"] = df_final["image_path"].str.replace(
            'dataset',
            str(self.root),
            n=1
        )

        def get_first_condition(list_string):
            try:
                conditions = ast.literal_eval(list_string)
                if conditions:
                    return conditions[0]
                else:
                    return None
            except (ValueError, SyntaxError):
                return None
        df_final['label'] = df_final['dermatologist_skin_condition_on_label_name'].apply(get_first_condition)
        df_final['label'] = df_final['label'].fillna("No finding")
        df_final['dermatologist_fitzpatrick_skin_type_label_1'] = df_final['dermatologist_fitzpatrick_skin_type_label_1'].fillna("Unknown")
        df_final["combined_race"] = df_final["combined_race"].fillna("Unknown")
        df_final["sex_at_birth"] = df_final["sex_at_birth"].fillna("Unknown")
        df_final["age_group"] = df_final["age_group"].fillna("Unknown")

        df_final.drop(571, inplace=True)

        self.df = df_final
        unique_labels = sorted(self.df['label'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_class = {k:v for v, k in self.class_to_idx.items()}

    @classmethod
    def get_info(cls):
        from src.dataloaders.dataset_registry import INFO
        return INFO["SCIN"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label_str = row["label"]
        label = self.class_to_idx[label_str]

        label = {
            "label": label,
            "label_str": label_str,
            "age": row["age_group"],
            "sex": row["sex_at_birth"],
            "race": row["combined_race"],
            "fitzpatrick": row["dermatologist_fitzpatrick_skin_type_label_1"],
        }

        return img, label