import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from src.backbones.mlp import MLP
from src.utils.embedding_cache_with_labels import EmbeddingCache, CacheConfig

@dataclass
class TrainingConfig:
    """
    Configuration class for the training script, implemented as a dataclass.
    
    This class holds all parameters needed to initialize the dataset,
    model, and training process.
    """
    
    datasets: List[str] = field(default_factory= lambda: ["cub"])
    data_root: str = "../data"
    model_path: str = "./ckpts/clip/openai-clip-vit-base-patch16-cub-best_model"
    cache_dir: str = "./cache"
    cls_only: bool = True
    layer_index: int = 10

    input_dim: int = 768
    num_classes: int = 200
    hidden_sizes: Optional[List[int]] = field(default_factory=lambda: [256, 128])
    drop_out: float = 0.1

    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3

    device: Optional[str] = "cuda"
    save_path: str = "ckpts/classifier/mlp.safetensors"

    def to_dict(self) -> dict:
        """
        Serializes the config object to a dictionary.
        This is used by CacheConfig.from_dict().
        """
        return asdict(self)


def train(cfg):
    device = torch.device(getattr(cfg, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Training on: {device}")

    cache_cfg = CacheConfig.from_dict(cfg.to_dict())
    cache = EmbeddingCache(cfg=cache_cfg)
    
    train_loader = DataLoader(cache.train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(cache.eval_dataset, batch_size=cfg.batch_size, shuffle=False) if cache.eval_dataset else None

    model = MLP(
        input_dim=cfg.input_dim, 
        num_classes=cfg.num_classes, 
        hidden_sizes=cfg.hidden_sizes, 
        dropout_rate=cfg.drop_out
    )
    print(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=getattr(cfg, 'learning_rate', 1e-3))
    epochs = getattr(cfg, 'epochs', 10)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}", end="")

        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            print(f" - Val Acc: {val_acc:.2f}%")
        else:
            print()

    save_path = getattr(cfg, 'save_path', './model.safetensors')
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    save_file(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    training_config = TrainingConfig(epochs=200)
    train(cfg=training_config)