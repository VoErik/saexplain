import torch
import wandb
from tqdm import tqdm
import numpy as np
import torch
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from dataclasses import dataclass, fields, field
from typing import Any, Dict
from src.data import get_dataset

@dataclass
class KnnEvalConfig:
    """Configuration for the k-NN classification evaluation."""

    # --- Model & Data Configuration ---
    model_type: str = "clip"
    pretrained_encoder_path: str = "./ckpts/clip/best_model"
    dataset_name: str = "fitzpatrick"
    data_root: str = "../../data"
    label_key: str = "label"
    num_classes: int = 114

    # --- k-NN Specific Parameters ---
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20, 40])

    # --- Hardware & Performance ---
    batch_size: int = 128
    num_workers: int = 4
    
    # --- Experiment Tracking ---
    wandb_project: str = "foundation_model_knn_eval"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "KnnEvalConfig":
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_config)

def get_all_embeddings(model, dataloader, device):
    """
    Helper function to compute and collect all embeddings from a dataloader.
    """
    all_embeddings = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            # Use the model's forward pass in inference mode to get CLS tokens
            cls_embeddings = model.forward(
                pixel_values=pixel_values,
                return_patch_embeddings=False
            )
            
            all_embeddings.append(cls_embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def evaluate_knn(config: dict):
    """
    Performs k-NN classification using Scikit-learn.
    """
    cfg = KnnEvalConfig.from_dict(config)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- Loading model and preparing data ---")
    if cfg.model_type.lower() == 'clip':
        from src.backbones import CLIP
        foundation_model = CLIP(cfg.pretrained_encoder_path).to(device)
    else:
        from src.backbones import MAE
        foundation_model = MAE(cfg.pretrained_encoder_path).to(device)

    processor = foundation_model.processor
    collate_fn = create_linear_probe_collate_fn(processor, label_key=cfg.label_key)

    from src.data import get_dataloaders
    train_loader, val_loader = get_dataloaders(
        datasets=cfg.dataset_name, data_root=cfg.data_root, transform=collate_fn,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, test_size=0.2
    )

    print("Computing training set embeddings...")
    train_embeddings, train_labels = get_all_embeddings(foundation_model, train_loader, device)

    print("Computing validation set embeddings...")
    val_embeddings, val_labels = get_all_embeddings(foundation_model, val_loader, device)
    
    del foundation_model

    train_embeddings_np = train_embeddings.numpy()
    train_labels_np = train_labels.numpy()
    val_embeddings_np = val_embeddings.numpy()
    val_labels_np = val_labels.numpy()

    for k in cfg.k_values:
        print(f"Fitting k-NN classifier with k={k}...")
        knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) # n_jobs=-1 uses all available CPU cores
        knn_classifier.fit(train_embeddings_np, train_labels_np)

        print("Predicting on the validation set...")
        predictions = knn_classifier.predict(val_embeddings_np)
        acc_score = accuracy_score(val_labels_np, predictions)

        print("\n--- k-NN Evaluation Results ---")
        print(f"k-NN Accuracy (k={k}): {acc_score:.4f}")
        wandb.log({f"knn_accuracy_k{k}": acc_score})
    wandb.finish()