import torch
from transformers import ViTImageProcessor, ViTMAEForPreTraining
from typing import Union, List, Optional, Tuple
from PIL import Image

from dataclasses import dataclass, fields
from typing import Any, Dict

import os
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from transformers import get_scheduler

class MAE(torch.nn.Module):
    """
    MAE model with a flexible forward pass that handles both raw data (for convenience)
    and pre-processed tensors (for speed), mirroring the provided CLIP class structure.
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTMAEForPreTraining.from_pretrained(model_name_or_path)
        self.encoder = self.model.vit
        self.num_layers = self.encoder.config.num_hidden_layers

    def forward(
        self,
        images: Optional[Union[List[Image.Image], torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        layer_index: int = -1,
        return_patch_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        A flexible forward pass that intelligently handles inputs.

        - For training (default): Pass `images` or `pixel_values`.
          Returns the full MAE model output, including reconstruction loss.
        - For inference: Set `return_patch_embeddings=True` or specify a `layer_index`.
          Returns the direct output of the ViT encoder (patch embeddings).
        """
        if pixel_values is None:
            if images is None:
                raise ValueError("You must provide either `images` or `pixel_values`.")
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values

        pixel_values = pixel_values.to(self.model.device)

        is_inference_mode = return_patch_embeddings or layer_index != -1
        
        if is_inference_mode:
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            all_hidden_states = encoder_outputs.hidden_states
            target_hidden_state = all_hidden_states[layer_index]
            
            cls_embedding = target_hidden_state[:, 0, :]
            
            if return_patch_embeddings:
                patch_embeddings = target_hidden_state[:, 1:, :]
                return cls_embedding, patch_embeddings
            else:
                return cls_embedding
        else:
            return self.model(pixel_values=pixel_values)
        

@dataclass
class MAETrainingConfig:
    # Data and Model Config
    dataset_name: str = "fitzpatrick"
    data_root: str = "../../data"
    model_name_or_path: str = "google/vit-base-patch16-224-in21k"
    output_dir: str = "./ckpts/mae"
    freeze_layers_up_to: Optional[int] = None

    # Training HPs
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1.5e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.05

    # Performance & Hardware
    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    
    # Experiment Tracking & Checkpointing
    wandb_project: str = "mae_pretraining_experiments"
    validation_interval: int = 1
    save_best_model: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MAETrainingConfig":
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_config)
    

import torch
from PIL import Image
from torchvision import transforms

class MAETransform:
    """
    A transform adapter for MAE pre-training or ViT feature extraction.
    Includes data augmentations.
    """
    def __init__(self, processor, is_training=True):
        """
        Args:
            processor (ViTImageProcessor): The processor for the ViT backbone.
            is_training (bool): If True, applies augmentations. If False,
                                only applies resizing and normalization (for validation/testing).
        """
        self.processor = processor
        self.is_training = is_training

        config = self.processor
        try:
            CROP_SIZE = config.size['height'] if isinstance(config.size, dict) else config.size
        except AttributeError:
             CROP_SIZE = config.crop_size['height'] if isinstance(config.crop_size, dict) else config.crop_size
             
        NORM_MEAN = config.image_mean
        NORM_STD = config.image_std

        if self.is_training:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=CROP_SIZE, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(size=CROP_SIZE),
                transforms.CenterCrop(size=CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])

    def __call__(self, raw_image: Image.Image):
        """
        Processes the raw PIL image using the defined transform pipeline.
        """
        processed_image = self.image_transform(raw_image)

        return {
            "pixel_values": processed_image
        }

def train_mae(config: dict):
    """Train MAE model, following the established CLIP training framework."""
    from src.data import get_dataloaders

    cfg = MAETrainingConfig.from_dict(config)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MAE(cfg.model_name_or_path).to(device)
    
    transform = MAETransform(processor=model.processor)

    train_loader, val_loader = get_dataloaders(
        datasets=cfg.dataset_name, data_root=cfg.data_root, transform=transform,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, test_size=0.2
    )

    if cfg.freeze_layers_up_to is not None:
        print(f"Freezing layers up to {cfg.freeze_layers_up_to}")
        for i, layer in enumerate(model.encoder.encoder.layers):
            if i < cfg.freeze_layers_up_to:
                for param in layer.parameters():
                    param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_training_steps = (len(train_loader) * cfg.num_epochs) // cfg.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)
    best_val_loss = float('inf')
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        pb = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for i, batch in enumerate(pb):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                outputs = model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
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
                    best_model_path = os.path.join(cfg.output_dir, f"{cfg.model_name_or_path.replace("/", "-")}-{cfg.dataset_name}-best_encoder")
                    model.encoder.save_pretrained(best_model_path)
                    model.processor.save_pretrained(best_model_path)
                    
    print("\nTraining finished.")
    final_model_path = os.path.join(cfg.output_dir, f"{cfg.model_name_or_path.replace("/", "-")}-{cfg.dataset_name}-final_encoder")
    model.encoder.save_pretrained(final_model_path)
    model.processor.save_pretrained(final_model_path)
    wandb.finish()


import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from typing import List, Optional, Tuple, Union

class MAEEncoder(torch.nn.Module):
    """
    A dedicated feature extractor that loads a pre-trained ViT encoder.
    """
    def __init__(self, model_path: str):
        super().__init__()
        # Load the processor and the ENCODER model
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTModel.from_pretrained(model_path)
        self.num_layers = self.model.config.num_hidden_layers

    def forward(
        self,
        images: Optional[Union[List[Image.Image], torch.Tensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        layer_index: int = -1, # Default to the last layer's output
        return_patch_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extracts features from the ViT encoder.

        Args:
            images: A list of PIL images or a batch of tensors.
            pixel_values: Pre-processed pixel values (alternative to `images`).
            layer_index: The index of the hidden layer to extract features from.
                         -1 corresponds to the last hidden layer.
            return_patch_embeddings: If True, returns both CLS and patch embeddings.

        Returns:
            - CLS token embedding(s) by default.
            - A tuple of (CLS embeddings, patch embeddings) if `return_patch_embeddings` is True.
        """
        if pixel_values is None:
            if images is None:
                raise ValueError("You must provide either `images` or `pixel_values`.")
            # The processor handles device placement if you pass `return_tensors="pt"`
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values

        # Ensure pixel_values are on the same device as the model
        pixel_values = pixel_values.to(self.model.device)

        # Get the encoder's outputs
        encoder_outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True # We need this to select a specific layer
        )
        
        # All hidden states from all layers are here
        all_hidden_states = encoder_outputs.hidden_states
        
        # Select the desired layer's output
        target_hidden_state = all_hidden_states[layer_index]
        
        # The CLS token is the first token of the sequence
        cls_embedding = target_hidden_state[:, 0, :]
        
        if return_patch_embeddings:
            # All other tokens are patch embeddings
            patch_embeddings = target_hidden_state[:, 1:, :]
            return cls_embedding, patch_embeddings
        else:
            return cls_embedding