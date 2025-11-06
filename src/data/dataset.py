import pandas as pd
import ast
import re

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Literal, List



import torch

# TODO: make general
class DermDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe, transform=None, labelkey='label'):
        super().__init__()
        self.df = dataframe
        self.transform = transform
        self.labelkey = labelkey

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]
        label = sample_data[self.labelkey]
        if isinstance(label, str) and label.startswith('['):
            try:
                label = ast.literal_eval(label)
            except (ValueError, SyntaxError, TypeError):
                label = ["error parsing prompt"]
        img = Image.open(sample_data["image_path"]).convert("RGB")

        if self.transform:
            transform_class_name = type(self.transform).__name__
            
            if transform_class_name == "MAETransform":
                return self.transform(img) 
            elif transform_class_name == "CLIPTransform":
                return self.transform(img, label)
            elif transform_class_name == "LinearProbeTransform":
                return self.transform(img, sample_data)
            else:
                return self.transform(img), label
        else:
            return img, label

CANONICAL_COLUMNS = [
    "source_dataset"
    "img_path",
    "label",
    "age",
    "sex",
    "location",
    "fitzpatrick_type",
    "clip_label"
]

def get_combined_dataset(
        names: List[Literal["scin", "midas", "fitzpatrick", "ham"]] = ["scin", "midas", "fitzpatrick", "ham"], 
        data_root='./data'
        ):
    dfs = []
    if "scin" in names:
        scin = setup_scin(root=str(Path(data_root) / "scin"))
        dfs.append(scin)
    if "midas" in names:
        midas = setup_midas(root=str(Path(data_root) / "mra-midas"))
        dfs.append(midas)
    if "fitzpatrick" in names:
        fp = setup_fitzpatrick(root=str(Path(data_root) / "fitzpatrick17k"))
        dfs.append(fp)
    if "ham" in names:
        ham = setup_ham10000(root=str(Path(data_root) / "ham10000"))
        dfs.append(ham)

    combined_df = pd.concat(dfs)
    return combined_df

def setup_fitzpatrick(root: str) -> pd.DataFrame:
    from src.backbones import generate_prompts_for_clip
    images_dir = Path(root) / "images"
    df = pd.read_csv(f"{root}/labels.csv")
    df["image_path"] = df["md5hash"].apply(lambda x: str(images_dir / f"{x}.jpg"))
    df.rename(columns={"fitzpatrick_scale": "fitzpatrick_type"}, inplace=True)
    df["clip_label"] = df.apply(generate_prompts_for_clip, axis=1)
    df["source_dataset"] = "fitzpatrick17k"
    df.drop(columns=["md5hash"], inplace=True)
    return df

def setup_ham10000(root: str) -> pd.DataFrame:
    from src.backbones import generate_prompts_for_clip
    images_dir = Path(root) / "images"
    df = pd.read_csv(f"{root}/HAM10000_metadata")
    df.rename(columns={"localization": "location", "dx": "label"}, inplace=True)
    df["image_path"] = df["image_id"].apply(lambda x: str(images_dir / f"{x}.jpg"))
    df["clip_label"] = df.apply(generate_prompts_for_clip, axis=1)
    df["source_dataset"] = "ham10000"
    df.drop(columns=["dataset", "image_id"], inplace=True)
    return df

def setup_scin(root: str) -> pd.DataFrame:
    """
    Loads and harmonizes the SCIN dataset, aggregating rich metadata.
    """
    from src.backbones import generate_prompts_for_clip
    LOC_COLS = [
            'body_parts_head_or_neck', 'body_parts_arm', 'body_parts_palm',
            'body_parts_back_of_hand', 'body_parts_torso_front',
            'body_parts_torso_back', 'body_parts_genitalia_or_groin',
            'body_parts_buttocks', 'body_parts_leg', 'body_parts_foot_top_or_side',
            'body_parts_foot_sole', 'body_parts_other'
        ]
    
    RACE_COLS = [
            'race_ethnicity_american_indian_or_alaska_native',
            'race_ethnicity_asian', 'race_ethnicity_black_or_african_american',
            'race_ethnicity_hispanic_latino_or_spanish_origin',
            'race_ethnicity_middle_eastern_or_north_african',
            'race_ethnicity_native_hawaiian_or_pacific_islander',
            'race_ethnicity_white', 'race_ethnicity_other_race'
        ]
    
    MORPH_COLS = [
            'textures_raised_or_bumpy', 'textures_flat', 
            'textures_rough_or_flaky', 'textures_fluid_filled'
        ]
    
    SYMPTOM_COLS = [
            'condition_symptoms_bothersome_appearance',
            'condition_symptoms_bleeding', 'condition_symptoms_increasing_size',
            'condition_symptoms_darkening', 'condition_symptoms_itching',
            'condition_symptoms_burning', 'condition_symptoms_pain',
            'other_symptoms_fever', 'other_symptoms_chills', 'other_symptoms_fatigue',
            'other_symptoms_joint_pain', 'other_symptoms_mouth_sores',
            'other_symptoms_shortness_of_breath'
        ]
    
    IGNORE_COLS = [
        'related_category', 'condition_duration', 'image_1_shot_type', 'image_2_shot_type', 
        'image_3_shot_type', 'combined_race', 'race_ethnicity_two_or_more_after_mitigation', 
        'dermatologist_gradable_for_skin_condition_1', 'dermatologist_gradable_for_skin_condition_2', 
        'dermatologist_gradable_for_skin_condition_3', 'dermatologist_skin_condition_on_label_name', 
        'dermatologist_skin_condition_confidence', 'weighted_skin_condition_label', 
        'dermatologist_gradable_for_fitzpatrick_skin_type_1', 'dermatologist_gradable_for_fitzpatrick_skin_type_2', 
        'dermatologist_gradable_for_fitzpatrick_skin_type_3', 'dermatologist_fitzpatrick_skin_type_label_1', 
        'dermatologist_fitzpatrick_skin_type_label_2', 'dermatologist_fitzpatrick_skin_type_label_3', 
        'gradable_for_monk_skin_tone_india', 'gradable_for_monk_skin_tone_us', 
        'monk_skin_tone_label_india', 'monk_skin_tone_label_us', 'race_ethnicity_prefer_not_to_answer',
        'year', 'release', 'condition_symptoms_no_relevant_experience', 'other_symptoms_no_relevant_symptoms',
        'source', 'sex_at_birth'
    ]

    def _aggregate_location(row):
        """Aggregates all 'body_parts_...' columns into a single string."""
        locations = []
        for col in LOC_COLS:
            if col in row and row[col] == True:
                # Cleans up the name, e.g., "body_parts_head_or_neck" -> "head or neck"
                locations.append(col.replace('body_parts_', '').replace('_', ' '))
        
        return ", ".join(locations) if locations else None # Return None if no location specified

    def _aggregate_morphology(row):
        """Aggregates all 'textures_...' columns into a single string."""
        
        textures = []
        for col in MORPH_COLS:
            if col in row and row[col] == True:
                textures.append(col.replace('textures_', '').replace('_', ' '))
                
        return ", ".join(textures) if textures else None # Return None if no texture specified
    
    def _aggregate_race(row):
        """Aggregates all 'race_ethnicity_...' columns into a single string."""
        
        races = []
        for col in RACE_COLS:
            if col in row and row[col] == True:
                # Cleans up the name, e.g., "race_ethnicity_asian" -> "asian"
                races.append(col.replace('race_ethnicity_', '').replace('_', ' '))
        
        if 'race_ethnicity_prefer_not_to_answer' in row and row['race_ethnicity_prefer_not_to_answer'] == True:
            return "Unknown"
            
        return ", ".join(races) if races else "Unknown"
    
    def _aggregate_symptoms(row) -> str | None:
        """Aggregates all specific symptom columns."""
        
        # Check for 'no relevant' columns first
        if (row.get('condition_symptoms_no_relevant_experience', False) or 
            row.get('other_symptoms_no_relevant_symptoms', False)):
            return None
            
        symptoms = []
        for col in SYMPTOM_COLS:
            if col in row and row[col] == True:
                # Cleans up name, e.g., "condition_symptoms_itching" -> "itching"
                desc = col.replace('condition_symptoms_', '').replace('other_symptoms_', '').replace('_', ' ')
                symptoms.append(desc)
                
        return ", ".join(symptoms) if symptoms else None

    labels_file = Path(root) / "scin_labels.csv"
    cases_file = Path(root) / "scin_cases.csv"

    cases_df = pd.read_csv(cases_file)
    cases_df['case_id'] = cases_df['case_id'].astype(str)
    labels_df = pd.read_csv(labels_file)
    labels_df['case_id'] = labels_df['case_id'].astype(str)
    merged_df = pd.merge(cases_df, labels_df, on='case_id')
    
    image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
    id_columns = [col for col in merged_df.columns if col not in image_path_columns]
    df_melted = pd.melt(merged_df, id_vars=id_columns, value_vars=image_path_columns,
                        var_name="original_cols", value_name="image_path")
    
    df_final = df_melted.dropna(subset=['image_path']).drop(columns='original_cols')
    
    df_final["image_path"] = df_final["image_path"].str.replace(
        'dataset', str(root), n=1
    )

    def get_first_condition(list_string):
        try:
            conditions = ast.literal_eval(list_string)
            return conditions[0] if conditions else "No finding"
        except (ValueError, SyntaxError):
            return "No finding"
            
    df_final['label'] = df_final['dermatologist_skin_condition_on_label_name'].apply(get_first_condition)
    df_final.rename(columns={"fitzpatrick_skin_type": "fitzpatrick_type"}, inplace=True)
    df_final["sex"] = df_final["sex_at_birth"].fillna("Unknown")
    df_final["age"] = df_final["age_group"].fillna("Unknown")
    df_final['location'] = df_final.apply(_aggregate_location, axis=1)
    df_final['morphology'] = df_final.apply(_aggregate_morphology, axis=1)
    df_final['race'] = df_final.apply(_aggregate_race, axis=1)
    df_final['symptoms'] = df_final.apply(_aggregate_symptoms, axis=1)

    df_final = df_final[df_final['image_path'].apply(lambda x: Path(x).exists())].reset_index(drop=True)
    df_final["clip_label"] = df_final.apply(generate_prompts_for_clip, axis=1)
    df_final["source_dataset"] = "scin"

    df_final.drop(columns=RACE_COLS, inplace=True)
    df_final.drop(columns=LOC_COLS, inplace=True)
    df_final.drop(columns=MORPH_COLS, inplace=True)
    df_final.drop(columns=SYMPTOM_COLS, inplace=True)
    df_final.drop(columns=IGNORE_COLS, inplace=True)

    return df_final

def setup_midas(root: str) -> pd.DataFrame:
    """
    Loads, harmonizes, and cleans the MRA-MIDAS dataset.
    """
    from src.backbones import generate_prompts_for_clip
    def is_image_valid(path):
        """Checks if an image file is not corrupted."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def _extract_midas_fitzpatrick_type(dataframe):
        """Extracts numeric Fitzpatrick type from the 'midas_fitzpatrick' column."""
        def get_fitzpatrick_number(text):
            if not isinstance(text, str):
                return None
            roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6'}
            match = re.match(r'^(vi|v|iv|iii|ii|i)\b', text.strip().lower())
            if match:
                return int(roman_map[match.group(0)])
            return None

        dataframe['fitzpatrick_type'] = dataframe['midas_fitzpatrick'].apply(get_fitzpatrick_number)
        return dataframe

    print("Setting up MRA-MIDAS dataset...")
    images_dir = Path(root) / "images"
    metadata_path = Path(root) / "release_midas.xlsx"

    df = pd.read_excel(metadata_path)

    df.rename(columns={
        'midas_path': 'label',
        'midas_age': 'age',
        'midas_gender': 'sex',
        'midas_race': 'race',
        'midas_location': 'location'
    }, inplace=True)
    
    df['label'] = df['label'].fillna("No finding").astype(str)
    df['age'] = df['age'].fillna("Unknown").astype(str)
    df['sex'] = df['sex'].fillna("Unknown").astype(str)
    df['race'] = df['race'].fillna("Unknown").astype(str)
    df['location'] = df['location'].fillna("Unknown").astype(str)
    
    df["midas_melanoma"] = df["midas_melanoma"].fillna("Unknown").astype(str)
    
    df['midas_file_name'] = df['midas_file_name'].str.replace('.jpeg', '.jpg', regex=False)

    df['image_path'] = df['midas_file_name'].apply(lambda x: str(images_dir / x)) # Store as string
    
    initial_count_exist = len(df)
    df = df[df['image_path'].apply(lambda x: Path(x).exists())].reset_index(drop=True)
    print(f"Found {len(df)} existing image files out of {initial_count_exist} total.")
    
    tqdm.pandas(desc="Checking image integrity")
    is_valid = df['image_path'].progress_apply(is_image_valid)
    
    corrupted_count = len(is_valid) - is_valid.sum()
    if corrupted_count > 0:
        print(f"Found and removed {corrupted_count} corrupted images.")
    
    df = df[is_valid].reset_index(drop=True)
    print(f"Dataset ready with {len(df)} valid images.")

    df = _extract_midas_fitzpatrick_type(df)

    df["clip_label"] = df.apply(generate_prompts_for_clip, axis=1)
    
    df["source_dataset"] = "mra_midas"

    master_schema_columns = [
        'image_path',
        'source_dataset',
        'clip_label',
        'label',
        'fitzpatrick_type',
        'sex',
        'age',
        'race',
        'location',
        'midas_melanoma'
    ]
    
    final_columns = [col for col in master_schema_columns if col in df.columns]
    df_clean = df[final_columns]
    
    print("MRA-MIDAS setup complete.")
    return df_clean