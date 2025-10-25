from dataclasses import dataclass, fields
from typing import Any, Dict, Callable

import torch
import torchmetrics
import wandb
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd # Make sure pandas is imported
import os


class LinearProbe(torch.nn.Module):
    def __init__(self, foundation_model, num_classes: int):
        """
        A linear probe model that uses the custom CLIP or MAE class as a frozen backbone.

        Args:
            foundation_model: An instance of our custom CLIP or MAE class.
            num_classes (int): The number of output classes for the linear head.
        """
        super().__init__()
        self.foundation_model = foundation_model
        
        for param in self.foundation_model.parameters():
            param.requires_grad = False
            
        if hasattr(self.foundation_model.model.config, 'vision_config'): # for CLIP
            feature_dim = self.foundation_model.model.config.vision_config.hidden_size
        else:
            feature_dim = self.foundation_model.model.config.hidden_size
            
        self.classifier_head = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor):
        self.foundation_model.eval()
        
        features = self.foundation_model.forward(
            pixel_values=pixel_values,
            layer_index=-1,
            return_patch_embeddings=False
        )
        
        logits = self.classifier_head(features)
        return logits
    


@dataclass
class LinearProbeConfig:
    """Configuration for the linear probing evaluation script."""

    # Model & Data Configuration
    model_type: str = "clip"  # Options: "clip" or "mae"
    pretrained_encoder_path: str = "./ckpts/clip/best_encoder"
    dataset_name: str = "fitzpatrick"
    data_root: str = "../../data"

    # Evaluation Task Configuration
    label_key: str = "label"
    num_classes: int = 114

    # Training Hyperparameters for the Linear Head
    learning_rate: float = 1e-3
    num_epochs: int = 25
    batch_size: int = 32
    
    #  Hardware & Performance
    num_workers: int = 4
    
    # Experiment Tracking
    wandb_project: str = "foundation_model_linear_probe"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LinearProbeConfig":
        """
        Instantiates the config from a dictionary, ignoring superfluous keys.
        """
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(**filtered_config)


import torch
import pandas as pd
from PIL import Image

class LinearProbeTransform:
    """
    Simplified transform for linear probing.
    Uses the provided processor for image handling.
    Maps string labels to integers using a provided mapping.
    """
    def __init__(self, processor, label_key: str, class_to_idx: dict):
        """
        Args:
            processor: Image processor from the foundation model.
            label_key (str): Column name in the DataFrame containing target STRING labels.
            class_to_idx (dict): Mapping from string labels to integer indices.
        """
        self.processor = processor
        self.label_key = label_key
        self.class_to_idx = class_to_idx

    def __call__(self, raw_image: Image.Image, sample_info: pd.Series):
        """
        Processes image using the processor and maps the string label to an integer index.
        """
        # 1. Process image
        image_inputs = self.processor(images=raw_image, return_tensors="pt")
        processed_image = image_inputs['pixel_values'].squeeze(0)

        # 2. Extract the STRING label and map it to an integer
        label_tensor = torch.tensor(-1, dtype=torch.long) # Default to ignore index
        try:
            label_str = sample_info[self.label_key]
            if pd.notna(label_str) and label_str in self.class_to_idx:
                 label_idx = self.class_to_idx[label_str]
                 label_tensor = torch.tensor(int(label_idx), dtype=torch.long)

        except KeyError:
             pass # Keep default -1

        return {
            "pixel_values": processed_image,
            "labels": label_tensor
        }


def evaluate_linear_probe(config: dict):
    """
    Trains and evaluates a linear probe. Maps string labels to integers.
    """
    cfg = LinearProbeConfig.from_dict(config)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading foundation model from: {cfg.pretrained_encoder_path}")
    if cfg.model_type.lower() == 'clip':
        from src.backbones import CLIP # Make sure this class exists
        foundation_model = CLIP(cfg.pretrained_encoder_path).to(device)
    elif cfg.model_type.lower() == 'mae':
        from src.backbones import MAEEncoder # Make sure this class exists
        foundation_model = MAEEncoder(cfg.pretrained_encoder_path).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    processor = foundation_model.processor

    DATA_ROOT = cfg.data_root
    from src.data.dataset import get_combined_dataset
    df = get_combined_dataset(names=cfg.dataset_name, data_root=DATA_ROOT)

    print(df.head())

    print(f"Mapping string labels from column '{cfg.label_key}' to integers...")
    unique_labels = sorted(df[cfg.label_key].unique())
    class_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_class = {v: k for k, v in class_to_idx.items()} # Optional: for logging
    num_classes = len(unique_labels)
    
    if cfg.num_classes != num_classes:
        print(f"Warning: Overriding config num_classes ({cfg.num_classes}) with actual count ({num_classes}).")
        cfg.num_classes = num_classes # Update config dataclass instance
        wandb.config.update({"num_classes": num_classes}, allow_val_change=True)


    label_idx_col = "label_idx"
    df[label_idx_col] = df[cfg.label_key].map(class_to_idx)
    print(f"Created integer label column '{label_idx_col}'. Num classes: {num_classes}")
    
    wandb.config.update({"class_mapping": idx_to_class})

    probe_transform = LinearProbeTransform(processor, label_key=cfg.label_key, class_to_idx=class_to_idx)
    from src.data import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        datasets=cfg.dataset_name,
        data_root=cfg.data_root,
        transform=probe_transform,
        labelkey=cfg.label_key,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    model = LinearProbe(foundation_model, num_classes=num_classes).to(device)

    # --- 6. Training Setup ---
    optimizer = AdamW(model.classifier_head.parameters(), lr=cfg.learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=-1)

    # --- Variables for Tracking Best Model ---
    best_val_accuracy = 0.0
    best_epoch = -1
    # Define where to save the best checkpoint (use wandb run dir for convenience)
    checkpoint_dir = os.path.join("ckpts", cfg.model_type, "linear_probes")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_linear_probe_head.pt")

    # Metrics (re-initialize per epoch for eval)
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.num_classes, ignore_index=-1).to(device)
    val_f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=cfg.num_classes, average="macro", ignore_index=-1).to(device)

    # --- 7. Training & **Epoch-wise Evaluation** Loop ---
    print("Starting linear probe training with epoch-wise evaluation...")
    for epoch in range(cfg.num_epochs):
        # --- Training Phase ---
        model.train()
        model.foundation_model.eval() # Keep backbone frozen
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]")
        total_train_loss = 0
        num_train_batches = 0
        for batch in progress_bar:
            valid_indices = batch["labels"] != -1
            if not valid_indices.any(): continue
            pixel_values = batch["pixel_values"][valid_indices].to(device)
            labels = batch["labels"][valid_indices].to(device)

            logits = model(pixel_values=pixel_values)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_train_loss += loss.item(); num_train_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval() # Set the probe head to eval mode
        val_accuracy_metric.reset() # Reset metrics at the start of epoch eval
        val_f1_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Val]", leave=False):
                valid_indices = batch["labels"] != -1
                if not valid_indices.any(): continue
                pixel_values = batch["pixel_values"][valid_indices].to(device)
                labels = batch["labels"][valid_indices].to(device)

                logits = model(pixel_values=pixel_values)
                preds = torch.argmax(logits, dim=1)
                val_accuracy_metric.update(preds, labels)
                val_f1_metric.update(preds, labels)

        # Compute metrics for the epoch
        current_val_acc = val_accuracy_metric.compute()
        current_val_f1 = val_f1_metric.compute()
        print(f"Epoch {epoch+1} Val Acc: {current_val_acc:.4f}, Val F1: {current_val_f1:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_probe_loss": avg_train_loss,
            "val_accuracy": current_val_acc,
            "val_macro_f1": current_val_f1,
        })

        # --- Save Best Model ---
        if current_val_acc > best_val_accuracy:
            best_val_accuracy = current_val_acc
            best_epoch = epoch + 1
            print(f"✨ New best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}. Saving model...")
            # Save ONLY the linear head's state dict
            torch.save(model.classifier_head.state_dict(), best_model_path)
            wandb.save(best_model_path) # Upload to wandb artifacts

    print(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.4f} achieved at epoch {best_epoch}.")

    # --- 8. Final Evaluation using Best Checkpoint ---
    print(f"Loading best model weights from epoch {best_epoch} for final evaluation...")
    if os.path.exists(best_model_path):
        model.classifier_head.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: Best model checkpoint not found. Evaluating with final epoch weights.")

    model.eval()
    val_accuracy_metric.reset()
    val_f1_metric.reset()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            # (Same evaluation loop as within the epoch loop)
            valid_indices = batch["labels"] != -1
            if not valid_indices.any(): continue
            pixel_values = batch["pixel_values"][valid_indices].to(device)
            labels = batch["labels"][valid_indices].to(device)
            logits = model(pixel_values=pixel_values)
            preds = torch.argmax(logits, dim=1)
            val_accuracy_metric.update(preds, labels)
            val_f1_metric.update(preds, labels)

    final_acc = val_accuracy_metric.compute()
    final_f1 = val_f1_metric.compute()

    print("\n--- Final Evaluation Results (Using Best Checkpoint) ---")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Macro F1-Score: {final_f1:.4f}")

    # Log final best metrics to wandb summary
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_accuracy"] = final_acc # Log the re-evaluated score
    wandb.summary["best_val_macro_f1"] = final_f1

    wandb.finish()