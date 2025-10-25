import os
import torch
import pandas as pd
import wandb
from tqdm import tqdm
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Union, Callable
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

@dataclass
class SubgroupEvalConfig:
    """Configuration for subgroup performance evaluation."""
    # Model & Data Confiug
    model_type: str = "clip"
    pretrained_encoder_path: str = "./ckpts/clip/best_model"
    dataset_name: Union[str, List[str]] = "fitzpatrick"
    data_root: str = "../../data"
    
    # Evaluation Task Config
    label_key: str = "label"
    num_classes: int = 114
    subgroup_columns: List[str] = field(default_factory=lambda: ["fp_scale"])

    # Training HPs for the Linear Head
    learning_rate: float = 1e-3
    num_epochs: int = 1
    batch_size: int = 256
    
    # Hardware & Performance
    num_workers: int = 4
    
    # Experiment Tracking
    wandb_project: str = "foundation_model_subgroup_eval"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SubgroupEvalConfig":
        """Instantiates the config from a dictionary, ignoring superfluous keys."""
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_config)



def create_subgroup_collate_fn(processor, label_key: str, metadata_keys: List[str]) -> Callable:
    """Creates a collate function that safely extracts specified metadata for subgroup analysis."""
    def collate_fn(batch: list) -> Dict:
        raw_images = [item[0] for item in batch]
        image_inputs = processor(images=raw_images, return_tensors="pt")
        labels = torch.stack([item[1][label_key] for item in batch])
        metadata = [{key: label_dict.get(key) for key in metadata_keys} for _, label_dict in batch]
        
        return {
            "pixel_values": image_inputs.pixel_values,
            "labels": labels,
            "metadata": metadata
        }
    return collate_fn

def evaluate_subgroups(config: dict):
    """Trains a linear probe and evaluates its performance across specified subgroups."""
    cfg = SubgroupEvalConfig.from_dict(config)
    run_name = f"{cfg.model_type}_{os.path.basename(cfg.pretrained_encoder_path)}_subgroup"
    wandb.init(project=cfg.wandb_project, config=vars(cfg), name=run_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading foundation model from: {cfg.pretrained_encoder_path}")
    if cfg.model_type.lower() == 'clip':
        from src.backbones import CLIP
        foundation_model = CLIP(cfg.pretrained_encoder_path).to(device)
    elif cfg.model_type.lower() == 'mae':
        from src.backbones import MAEEncoder
        foundation_model = MAEEncoder(cfg.pretrained_encoder_path).to(device)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    from src.eval.linear_probe import LinearProbe
    model = LinearProbe(foundation_model, num_classes=cfg.num_classes).to(device)
    
    processor = foundation_model.processor
    train_collate_fn = create_subgroup_collate_fn(processor, cfg.label_key, [])
    eval_collate_fn = create_subgroup_collate_fn(processor, cfg.label_key, cfg.subgroup_columns)

    from src.data import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        datasets=cfg.dataset_name, data_root=cfg.data_root, transform=train_collate_fn,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, test_size=0.2
    )
    val_loader.collate_fn = eval_collate_fn

    optimizer = AdamW(model.classifier_head.parameters(), lr=cfg.learning_rate)
    loss_fn = CrossEntropyLoss()
    
    print("Starting linear probe training...")
    for epoch in range(cfg.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values=pixel_values)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print("\nTraining finished. Evaluating on validation set for subgroup analysis...")
    model.eval()
    
    all_preds, all_labels, all_metadata = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating for Subgroups"):
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
            all_metadata.extend(batch["metadata"])

    print("\n--- Subgroup Performance Analysis ---")
    results_df = pd.DataFrame(all_metadata)
    results_df['prediction'] = all_preds
    results_df['true_label'] = all_labels
    
    overall_accuracy = accuracy_score(results_df['true_label'], results_df['prediction'])
    print(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    wandb.log({"eval_overall_accuracy": overall_accuracy})

    for col in cfg.subgroup_columns:
        print(f"--- Performance by {col} ---")
        subgroup_acc = results_df.groupby(col).apply(
            lambda g: accuracy_score(g['true_label'], g['prediction'])
        ).to_dict()
        
        for group, acc in subgroup_acc.items():
            print(f"{group}: {acc:.4f}")
        
        wandb.log({f"eval_accuracy_by_{col}": subgroup_acc})
        print("-" * 25)

    wandb.finish()