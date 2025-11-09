import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any
from pathlib import Path
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from safetensors.torch import save_file

from src.utils.load_backbone import load_backbone
from sae.utils.misc import filter_valid_dataclass_fields
from src.backbones.mlp import MLP
from src.data import get_dataloaders

@dataclass
class BackboneTrainingConfig:
    architecture: str
    pretrained: bool = True
    model_name: str | None = None
    checkpoint_path: str | None = None
    is_train: bool = True
    hidden_sizes: List[int] = field(default_factory=list)
    num_classes: int = 200
    dropout_rate: float = 0.0
    freezing_mode: Literal['frozen', 'partial', 'unfrozen'] = 'frozen'
    num_unfrozen_blocks: int = 0 
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone_save_path: str = "./ckpts/"
    head_save_path: str = f"./ckpts/classifier/"
    data_root: str = "../../data"
    datasets: list[str] = field(default_factory=lambda: ["cub"])

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res

class ViTClassifier(nn.Module):
    def __init__(self, cfg: BackboneTrainingConfig):
        super().__init__()
        self.cfg = cfg
        
        self.backbone, self.train_transform, self.val_transform = load_backbone(
            architecture=cfg.architecture,
            model_name=cfg.model_name,
            is_train=cfg.is_train,
            checkpoint_path=cfg.checkpoint_path,
        )

        self.head = MLP(
            input_dim=self.backbone.embedding_dim,
            num_classes=cfg.num_classes,
            hidden_sizes=cfg.hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            use_input_norm=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        x = features.cls_embedding
        x = self.head(x)
        return x
    

    def save(self, backbone_path: str, head_path: str):
        """
        Splits the model into two safetensor files:
        1. backbone_path: generic timm weights.
        2. head_path: task-specific weights (LayerNorm + MLP head).
        """
        self.backbone.save(backbone_path)

        head_tensors = {
            k: v for k, v in self.state_dict().items() 
            if k.startswith('norm.') or k.startswith('head.')
        }

        os.makedirs(os.path.dirname(head_path) or '.', exist_ok=True)
        save_file(head_tensors, head_path)
    
def setup_optimizer_and_freezing(model: ViTClassifier, cfg: BackboneTrainingConfig) -> torch.optim.Optimizer:
    """
    Configures freezing strategies and returns an optimizer with appropriate parameter groups.
    """
    mode = cfg.freezing_mode
    
    if mode == 'frozen':
        print("Strategy: Freezing ENTIRE backbone.")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    elif mode == 'partial':
        print(f"Strategy: Partial freezing. Tuning last {cfg.num_unfrozen_blocks} blocks.")
        if hasattr(model.backbone, 'freeze_all_except_last_n_blocks'):
             model.backbone.freeze_all_except_last_n_blocks(cfg.num_unfrozen_blocks)
        else:
             raise AttributeError("Backbone does not support 'freeze_all_except_last_n_blocks'")
             
    elif mode == 'unfrozen':
        print("Strategy: Backbone completely UNFROZEN.")
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    backbone_params = filter(lambda p: p.requires_grad, model.backbone.parameters())
    
    head_params_list = list(model.head.parameters())
    head_params = filter(lambda p: p.requires_grad, head_params_list)

    param_groups = [
        {'params': backbone_params, 'lr': cfg.lr_backbone},
        {'params': head_params, 'lr': cfg.lr_head}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer


def train_backbone(config: dict):
    cfg = BackboneTrainingConfig.from_dict(config)
    device = torch.device(cfg.device)
    model = ViTClassifier(cfg).to(device)

    backbone_save_path = f"{cfg.architecture}/{model.backbone.name.replace("/", "-")}-{"-".join(cfg.datasets) if isinstance(cfg.datasets, list) else cfg.datasets}.safetensors"
    head_save_path = f"classifiers/head-{model.backbone.name.replace("/", "-")}-{"-".join(cfg.datasets) if isinstance(cfg.datasets, list) else cfg.datasets}.safetensors"
    
    optimizer = setup_optimizer_and_freezing(model, cfg)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(
            datasets=cfg.datasets, # type: ignore
            data_root=cfg.data_root, 
            train_transform=model.train_transform,
            val_transform=model.val_transform, 
            batch_size=64,
            num_workers=4,
        )
    
    best_val_acc = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(
                backbone_path=str(Path(cfg.backbone_save_path) / backbone_save_path), 
                head_path=str(Path(cfg.head_save_path) / head_save_path)
            )
            print(f" >>> New best model saved! (Acc: {best_val_acc:.2f}%)")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total