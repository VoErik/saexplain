import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
import wandb
from PIL import Image
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor, get_scheduler
from src.data import get_dataloaders
import pandas as pd
import numpy as np
import random

class CLIP(torch.nn.Module):
    """
    CLIP model with a flexible forward pass that handles
    both raw data (for convenience) and pre-processed tensors (for speed).
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.num_layers = self.model.vision_model.config.num_hidden_layers

    def forward(
        self,
        images: Optional[Union[List[Image.Image], torch.Tensor]] = None,
        texts: Optional[Union[List[str], torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_index: int = -1,
        return_patch_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        A flexible forward pass that intelligently handles inputs.

        - For convenient inference, pass raw `images` and/or `texts`.
        - For efficient training, pass pre-processed tensors: `pixel_values`, `input_ids`, and `attention_mask`.
        """

        if pixel_values is None:
            if images is None:
                raise ValueError("You must provide either `images` or `pixel_values`.")
            image_inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = image_inputs.pixel_values

        if input_ids is None and texts is not None:
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

        pixel_values = pixel_values.to(self.model.device)
        if input_ids is not None:
            input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # if no text is provided --> inference mode
        is_training_mode = input_ids is not None
        
        if not is_training_mode:
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            all_hidden_states = vision_outputs.hidden_states
            target_hidden_state = all_hidden_states[layer_index]
            
            cls_embedding = target_hidden_state[:, 0, :] # CLS token is the first token
            
            if return_patch_embeddings:
                return target_hidden_state[:, :, :]
            else:
                return cls_embedding
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
        
@dataclass
class CLIPTrainingConfig:
    # Data and Model Config
    dataset_name: str = "fitzpatrick"
    data_root: str = "../../data"
    model_name_or_path: str = "openai/clip-vit-base-patch32"
    output_dir: str = "./ckpts/clip"
    freeze_layers_up_to: Optional[int] = None 

    # Training HPs
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # Performance & Hardware
    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    
    # Experiment Tracking & Checkpointing
    wandb_project: str = "clip_finetuning_experiments"
    validation_interval: int = 1
    save_best_model: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CLIPTrainingConfig":
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_config)
    
def train_clip(config: dict):
    """Train CLIP model."""

    cfg = CLIPTrainingConfig.from_dict(config)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIP(cfg.model_name_or_path).to(device)
    
    transform = CLIPTransform(model.processor)
    train_loader, val_loader = get_dataloaders(
        datasets=cfg.dataset_name,
        data_root=cfg.data_root, 
        transform=transform, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        test_size=0.2,
        labelkey="clip_label"
    )

    if cfg.freeze_layers_up_to is not None:
        print(f"Freezing layers up to {cfg.freeze_layers_up_to}")
        for i, layer in enumerate(model.model.vision_model.encoder.layers):
            if i < cfg.freeze_layers_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_training_steps = (len(train_loader) * cfg.num_epochs) // cfg.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)
    
    best_val_loss = float('inf')
    os.makedirs(cfg.output_dir, exist_ok=True)
    

    for epoch in range(cfg.num_epochs):
        model.train()
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for i, batch in enumerate(pb):
            batch = {k: v.to(device) for k, v in batch.items()}

            if epoch == 0 and i == 0:
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Pixel values shape: {batch['pixel_values'].shape}")
                if 'input_ids' in batch:
                    print(f"Input IDs shape: {batch['input_ids'].shape}")
                if 'attention_mask' in batch:
                    print(f"Attention mask shape: {batch['attention_mask'].shape}")
                print("Starting training...\n")
            
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            wandb.log({"train_loss": loss.item() * cfg.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]})

        if epoch % cfg.validation_interval == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                        outputs = model(**batch)
                        total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
            print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                if cfg.save_best_model:
                    best_model_path = os.path.join(cfg.output_dir, f"{cfg.model_name_or_path.replace("/", "-")}-{cfg.dataset_name}-best_model")
                    model.model.save_pretrained(best_model_path)
                    model.processor.save_pretrained(best_model_path)
                    
    print("\nTraining finished.")
    final_model_path = os.path.join(cfg.output_dir, f"{cfg.model_name_or_path.replace("/", "-")}-{cfg.dataset_name}-final_model")
    model.model.save_pretrained(final_model_path)
    model.processor.save_pretrained(final_model_path)
    wandb.finish()


from torchvision import transforms
import random
from PIL import Image

class CLIPTransform:
    """
    A transform "adapter" for the CLIP model.
    It applies custom image augmentations and processes both image and text.
    """
    def __init__(self, processor):
        self.processor = processor
        
        CROP_SIZE = self.processor.image_processor.crop_size['height']
        NORM_MEAN = self.processor.image_processor.image_mean
        NORM_STD = self.processor.image_processor.image_std
        
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=CROP_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

    def __call__(self, raw_image: Image.Image, raw_prompts_list: list[str]):
        processed_image = self.image_transform(raw_image)
        selected_prompt = random.choice(raw_prompts_list)
        
        processed_text = self.processor(
            text=selected_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        
        return {
            "pixel_values": processed_image,
            "input_ids": processed_text['input_ids'].squeeze(0), # Remove batch dim
            "attention_mask": processed_text['attention_mask'].squeeze(0) # Remove batch dim
        }



# --- Expanded Base Templates for Pathologies ---
BASE_TEMPLATES = [
    # Standard
    "a photo of a {label}.",
    "an image of a {label}.",
    "a close-up of a {label} on the skin.",
    # Clinical Framing
    "a dermatological image showing a {label}.",
    "a clinical presentation of a {label}.",
    "an example of {label} in a clinical setting.",
    "a case study image of {label}.",
    # Descriptive
    "skin affected by {label}.",
    "a finding consistent with {label}.",
    "an area of skin showing signs of {label}."
]

# --- Expanded Healthy/Negative Templates ---
HEALTHY_TEMPLATES = [
    # Standard
    "a photo of healthy skin.",
    "an image of normal, unaffected skin.",
    "a dermatological image with no significant findings.",
    # Clinical Framing
    "a clinical photo showing clear skin.",
    "an unremarkable area of skin.",
    "a photo showing healthy skin tissue.",
    "skin with no evidence of pathology."
]

# (No changes needed here, but kept for context)
NEGATIVE_LABELS = ['no finding', 'healthy', 'normal skin', 'unremarkable']

COMPONENT_PHRASES = {
    'fitzpatrick_type': [
        "on Fitzpatrick skin type {fitzpatrick_type}",
        "on skin type {fitzpatrick_type}",
        "as it appears on type {fitzpatrick_type} skin",
        "in an individual with Fitzpatrick type {fitzpatrick_type} skin",
        "presenting on skin classified as Fitzpatrick {fitzpatrick_type}"
    ],
    'location': [
        "on the {location}",
        "located on the patient's {location}",
        "found on the {location}",
        "affecting the skin of the {location}",
        "appearing on the {location} area"
    ],
    'age': [
        "on a {age}-year-old patient",
        "affecting a person of age {age}",
        "in a patient who is {age} years old",
        "observed in a {age}-year-old individual"
    ],
    'gender': [
        "on a {gender} patient",
        "in a {gender}",
        "observed in a {gender} individual",
        "a case affecting a {gender}"
    ],
}


DISEASE_NAME_MAP = {
            "bkl": "benign keratosis-like lesion",
            "nv": "melanocytic nevus",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "vasc": "vascular lesion",
            "bcc": "basal cell carcinoma",
            "akiec": "actinic keratosis"
        }

def format_age_phrase(age_data) -> str | None:
    """Formats an age phrase based on whether the input is numeric or a string."""
    if isinstance(age_data, (int, float)) and 0 < age_data < 120:
        return f"on a {int(age_data)}-year-old patient"
    if isinstance(age_data, (str)) and age_data.isdigit():
        return f"on a {int(age_data)}-year-old person"
    elif isinstance(age_data, str) and age_data.lower() not in ['unknown', 'na']:
        # Clean up string-based age groups (e.g., "AGE_18_TO_25" -> "18 to 25")
        clean_age = age_data.upper().replace('AGE_', '').replace('_', ' ').lower()
        return f"from a patient in the {clean_age} age group"
    return None

def format_fitzpatrick_phrase(fitz_data) -> str | None:
    """Formats a Fitzpatrick phrase based on whether the input is numeric or a string."""
    if isinstance(fitz_data, (int, float)) and 1 <= fitz_data <= 6:
        return f"on Fitzpatrick skin type {int(fitz_data)}"
    elif isinstance(fitz_data, str) and fitz_data.lower() not in ['unknown', 'na']:
        return f"on skin described as '{fitz_data.lower()}'"
    return None

def format_gender_phrase(gender_data) -> str | None:
    """Formats a Gender phrase based on whether the input is numeric or a string."""
    if isinstance(gender_data, (str)) and gender_data.lower() not in ['unknown', 'na', 'other', 'other_or_unspecified']:
        return f"on a {gender_data.lower()} patient"
    return None

def format_location_phrase(location_data) -> str | None:
    """Formats a location phrase."""
    if isinstance(location_data, str) and location_data.lower() not in ['unknown', 'na']:
        return f"located on the {location_data.lower()}"
    return None

def generate_prompts_for_clip(row: pd.Series) -> list[str]:
    """
    Orchestrates prompt generation by calling specialized helper functions
    and combining their outputs.
    """
    label = row.get('label', 'skin condition')
    if isinstance(label, torch.Tensor):
        label = label.item()

    if label in DISEASE_NAME_MAP:
        label = DISEASE_NAME_MAP[str(label)]

    if isinstance(label, str) and label.lower() in NEGATIVE_LABELS:
        return list(HEALTHY_TEMPLATES)

    component_phrases = []
    
    age_phrase = format_age_phrase(row.get('age'))
    if age_phrase: component_phrases.append(age_phrase)

    fitz_phrase = format_fitzpatrick_phrase(row.get('fitzpatrick_type'))
    if fitz_phrase: component_phrases.append(fitz_phrase)

    location_phrase = format_location_phrase(row.get('location'))
    if location_phrase: component_phrases.append(location_phrase)

    gender_phrase = format_gender_phrase(row.get('sex'))
    if gender_phrase: component_phrases.append(gender_phrase)

    prompts = [template.format(label=label.lower()) for template in BASE_TEMPLATES]

    for phrase in component_phrases:
        prompts.append(f"A clinical image of a {label.lower()}, {phrase}.")

    if len(component_phrases) >= 2:
        phrase1, phrase2 = random.sample(component_phrases, 2)
        prompts.append(f"A case of {label.lower()} {phrase1}, {phrase2}.")

    if len(component_phrases) >= 3:
        phrase1, phrase2, phrase3 = random.sample(component_phrases, 3)
        prompts.append(f"A case of {label.lower()} {phrase1}, {phrase2}, {phrase3}.")

    return list(set(prompts)) # Use set to remove any potential duplicates