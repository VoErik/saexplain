import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from tqdm import tqdm
import time
import wandb
from dataclasses import dataclass, field
from pathlib import Path
import torchvision.transforms as T
import pandas as pd
from src.dataloaders import ImageDataManager

def get_fp_classweights(device, labeltype):
    labels_df = pd.read_csv("data/fitzpatrick17k/labels.csv")
    class_counts = labels_df[labeltype].value_counts().sort_index()
    class_weights = len(labels_df) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
    return class_weights_tensor

def get_transforms(img_size: int, mode: str):
    """
    Returns a dictionary of standard transforms for training and validation.
    Uses ImageNet normalization stats as is standard for transfer learning.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms = T.Compose([
                    T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=(-15,15)),
                    T.ToTensor(),
                    T.Normalize(mean=imagenet_mean, std=imagenet_std)
                ])

    return transforms

@dataclass
class ResNetTrainerConfig:
    # Core settings
    batch_size: int = 64
    learning_rate: float = 5e-4 * batch_size / 256
    epochs: int = 100
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_key: str = "label"

    # Optimizer and Scheduler
    optimizer_name: str = "AdamW"
    weight_decay: float = 0.05
    scheduler_name: str = "CosineAnnealingLR"
    warmup_epochs: int = 5

    # Regularization
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10

    # Loss Function
    loss_fn: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1, weight=get_fp_classweights(device, label_key))

    # Dataset and Model
    dataset: str = "fitzpatrick17k"
    data_root: str = "data"
    output_dir: Path = Path("./output")
    seed: int = 42
    img_size: int = 224
    num_workers: int = 8
    num_classes: int = 114
    arch: str = "resnet18"

    # Wandb
    wandb_project: str = "resnet-baseline"
    wandb_team: str = "voerik"


class ResNetTrainer:
    """Trainer for ResNet."""
    def __init__(self, cfg: ResNetTrainerConfig):
        self.cfg = cfg
        self.device = cfg.device

        print("Setting up data...")
        self.datamanager = ImageDataManager(
            data_root=self.cfg.data_root,
            initialize=[self.cfg.dataset],
            seed=self.cfg.seed,
            transform=get_transforms(self.cfg.img_size, mode="train")
        )
        self.train_loader, self.val_loader = self._get_dataloaders()
        self.num_classes = self.cfg.num_classes

        print("Setting up model...")
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.loss_fn = self.cfg.loss_fn.to(self.device)
        self.global_step = 0

    def _get_model(self):
        if self.cfg.arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.cfg.arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError("Model architecture not recognized: {}".format(self.cfg.arch))
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )

        return model

    def _get_dataloaders(self):
        return self.datamanager.get_dataloaders(
            dataset=self.cfg.dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            test_size=0.15
        )

    def _get_optimizer(self):
        if self.cfg.optimizer_name == "AdamW":
            return optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer_name}")

    def _get_scheduler(self):
        if self.cfg.scheduler_name == "CosineAnnealingLR":
            return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs - self.cfg.warmup_epochs)
        else:
            return None

    def _run_validation(self):
        self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels_dict in self.val_loader:
                images = images.to(self.device)
                labels = labels_dict[self.cfg.label_key].to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(self.val_loader), correct / total_samples

    def train(self):
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{self.cfg.dataset}_{self.cfg.arch}_{int(time.time())}"
        wandb.init(project=self.cfg.wandb_project, name=run_name, config=vars(self.cfg), entity=self.cfg.wandb_team)

        best_val_accuracy = -1.0
        patience_counter = 0
        total_warmup_steps = self.cfg.warmup_epochs * len(self.train_loader)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_train_loss = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
            for i, (images, labels_dict) in enumerate(pbar):
                # --- Warmup ---
                current_step = epoch * len(self.train_loader) + i
                if self.cfg.warmup_epochs > 0 and current_step < total_warmup_steps:
                    lr_scale = (current_step + 1) / total_warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.cfg.learning_rate * lr_scale

                images = images.to(self.device)
                labels = labels_dict[self.cfg.label_key].to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
                self.optimizer.step()

                total_train_loss += loss.item()
                wandb.log({
                    "train/loss_step": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "global_step": self.global_step
                })
                self.global_step += 1
                pbar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss, val_accuracy = self._run_validation()

            if self.scheduler and epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.cfg.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            wandb.log({
                "train/avg_loss_epoch": avg_train_loss,
                "val/avg_loss_epoch": avg_val_loss,
                "val/accuracy": val_accuracy,
                "epoch": epoch + 1
            })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(self.model.state_dict(), self.cfg.output_dir / f"best_{self.cfg.arch}.pth")
                print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"Stopping early. Validation accuracy has not improved for {self.cfg.early_stopping_patience} epochs.")
                    break

        wandb.finish()
        print("\nTraining finished.")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")