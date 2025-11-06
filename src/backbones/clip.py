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
import random
import torch.nn as nn

class CLIP(torch.nn.Module):
    """
    CLIP model with a flexible forward pass that handles
    both raw data (for convenience) and pre-processed tensors (for speed).
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path) # use original img processor
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
            final_cls_embedding = all_hidden_states[-1][:, 0, :]
            
            if return_patch_embeddings:
                return target_hidden_state[:, :, :], all_hidden_states[-1][:, :, :]
            else:
                return cls_embedding, final_cls_embedding
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )

class CLIPForClassification(nn.Module):
    """
    A classification model that uses the CLIP vision encoder as a backbone.
    """
    def __init__(self, model_name_or_path: str, num_classes: int, layer_index: int = -1):
        """
        Args:
            model_name_or_path (str): Path to the pretrained CLIP model.
            num_classes (int): The number of output classes (e.g., 200 for CUB).
            layer_index (int): Which hidden state layer to use for the CLS token.
                               -1 (default) uses the final hidden state.
                               0 uses the initial embedding.
                               1 to N uses the Nth transformer layer output.
        """
        super().__init__()
        self.num_classes = num_classes
        self.layer_index = layer_index
        
        self.clip = CLIP(model_name_or_path)
        
        embed_dim = self.clip.model.vision_model.config.hidden_size
        
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        """
        
        cls_embedding, _ = self.clip(
            pixel_values=pixel_values,
            layer_index=self.layer_index 
        )
        
        logits = self.classification_head(cls_embedding)
        
        return logits

@dataclass
class CLIPTrainingConfig:
    # Data and Model Config
    dataset_name: str = "cub"
    data_root: str = "../../data"
    model_name_or_path: str = "openai/clip-vit-base-patch16"
    output_dir: str = "./ckpts/clip"
    freeze_layers_up_to: Optional[int] = None 

    # Training HPs
    num_epochs: int = 10
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
    
    transform = CLIPTransform(model.processor, is_train=True)
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
    def __init__(self, processor, is_train):
        self.processor = processor
        self.is_train = is_train

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

    def __call__(self, raw_image: Image.Image, raw_prompts_list: list[str] | None = None):
        if self.is_train:
            processed_image = self.image_transform(raw_image)
            selected_prompt = random.choice(raw_prompts_list)
        
            processed_text = self.processor(
                text=selected_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77
            )
        else:
            out = self.processor(
                images=raw_image,
                return_tensors="pt",
                padding="True")
            
            processed_image = out["pixel_values"].squeeze(0)

            return {"pixel_values": processed_image}
        
        
        return {
            "pixel_values": processed_image,
            "input_ids": processed_text['input_ids'].squeeze(0), # Remove batch dim
            "attention_mask": processed_text['attention_mask'].squeeze(0) # Remove batch dim
        }

def train_one_epoch(
    model: CLIPForClassification,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_accumulation_steps: int
):
    model.train()
    total_loss = 0.0
    
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        # --- Adapt this to your dataset ---
        # Assuming batch is a dict with 'pixel_values' and 'labels'
        pixel_values = images["pixel_values"].to(device)
        labels = labels.to(device)
        # ------------------------------------

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(pixel_values=pixel_values)
            loss = criterion(logits, labels)
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(
    model: CLIPForClassification,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool
):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            pixel_values = images["pixel_values"].to(device)
            labels = labels.to(device)
            # ------------------------------------

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(pixel_values=pixel_values)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = (all_preds == all_labels).float().mean().item()
    
    return avg_loss, accuracy

def freeze_clip_layers(model: CLIPForClassification, freeze_layers_up_to: int):
    """
    Freezes the CLIP vision backbone according to the config.
    """
    if freeze_layers_up_to is None or freeze_layers_up_to < 0:
        print("No layers frozen. Fine-tuning the entire vision model.")
        return

    # The vision model is at model.clip.model.vision_model
    vision_model = model.clip.model.vision_model
    
    # 1. Freeze the patch and position embeddings
    for param in vision_model.embeddings.parameters():
        param.requires_grad = False

    # 2. Freeze the 'pre_layrnorm'
    if hasattr(vision_model, 'pre_layrnorm'):
         for param in vision_model.pre_layrnorm.parameters():
            param.requires_grad = False

    # 3. Freeze the specified number of transformer layers
    if freeze_layers_up_to > 0:
        for i, layer in enumerate(vision_model.encoder.layers[:freeze_layers_up_to]):
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Froze vision layer {i}")

    print(f"Successfully froze embeddings and {freeze_layers_up_to} vision layers.")

def run_training_classification(cfg: dict, num_classes: int):
    
    config = CLIPTrainingConfig.from_dict(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Initialize WandB
    wandb.init(project=config.wandb_project, config=vars(config))

    # 2. Initialize Model
    model = CLIPForClassification(
        model_name_or_path=config.model_name_or_path,
        num_classes=num_classes
    ).to(device)

    transform = CLIPTransform(model.clip.processor, is_train=False)
    train_loader, val_loader = get_dataloaders(
        datasets=config.dataset_name,
        data_root=config.data_root, 
        transform=transform, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers,
        test_size=0.2,
        labelkey="clip_label"
    )
    
    # 3. Apply Layer Freezing
    freeze_clip_layers(model, config.freeze_layers_up_to)

    # 4. Initialize Optimizer
    # Separate params for different LRs if needed, but this is a good start
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # Only optimize unfrozen params
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 5. Initialize Loss Function and Scaler
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    # 6. Initialize LR Scheduler
    num_training_steps = config.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0, # You can add a warmup_steps to your config
        num_training_steps=num_training_steps,
    )

    # 7. Training Loop
    best_val_accuracy = 0.0

    for epoch in range(config.num_epochs):
        print(f"--- Epoch {epoch + 1} / {config.num_epochs} ---")
        
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=config.use_amp,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        if (epoch + 1) % config.validation_interval == 0:
            val_loss, val_accuracy = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=config.use_amp
            )
            
            print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if config.save_best_model:
                    best_model_path = os.path.join(config.output_dir, f"{config.model_name_or_path.replace("/", "-")}-{config.dataset_name}-best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    model.clip.model.save_pretrained(best_model_path)
                    model.clip.processor.save_pretrained(best_model_path)
                    
                    print(f"New best model saved in Hugging Face format to {best_model_path}")

    wandb.finish()
    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")


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