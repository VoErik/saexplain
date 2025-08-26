from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import time
import itertools
import wandb
from dataclasses import dataclass, field

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.dataloaders import ImageDataManager, get_transforms
from src.models.cbm import CBM, CBMConfig


@dataclass
class CBMTrainerConfig:
    # Core settings
    batch_size: int = 32
    learning_rate: float = 5e-4 * batch_size / 256
    concept_epochs: int = 500
    classifier_epochs: int = 500
    joint_epochs: int = 1000
    lambda_weight: float = 0.1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label: str = "label" # Key for the task label in the data dictionary

    # Optimizer and Scheduler
    optimizer_name: str = "Adam"
    scheduler_name: str = "CosineAnnealingLR"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {"T_max": 500})
    gradient_clip_val: float = 1.0
    warmup_epochs: int = 20
    warmup_lr_init: float = 1e-6

    # Early Stopping
    early_stopping_patience: int = 25

    # Loss Functions
    concept_loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    classifier_loss_fn: nn.Module = nn.CrossEntropyLoss()

    # Dataset and Model
    training_datasets: list = field(default_factory=lambda: ["skincon_fitzpatrick17k"])
    dataset: str = "skincon_fitzpatrick17k"
    data_root: str = "data"
    shuffle: bool = True
    validation_split: float = 0.15
    output_dir: Path = Path("./output")
    cbm_config: CBMConfig = field(default_factory=CBMConfig)
    freeze_feature_extractor: bool = False
    seed: int = 42
    img_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True

    # Wandb
    wandb_team: str = "voerik"
    wandb_project: str = "cbm-training-sophisticated"
    wandb_name: str = "cbm-run"


class CBMTrainer:
    """Trainer for Concept Bottleneck Models."""
    def __init__(self, cfg: CBMTrainerConfig):
        self.cfg = cfg
        self.model = self._get_model()
        self.datamanager = ImageDataManager(
            data_root=self.cfg.data_root,
            initialize=self.cfg.training_datasets,
            seed=self.cfg.seed,
            transform=get_transforms(self.cfg.img_size)
        )
        self.train_loader, self.val_loader = self._get_dataloaders()
        self.task_loss_fn = self.cfg.classifier_loss_fn
        self.concept_loss_fn = self.cfg.concept_loss_fn
        self.device = self.cfg.device
        self.global_step = 0
        self.model.to(self.device)

    def _get_model(self):
        return CBM(self.cfg.cbm_config)

    def _get_dataloaders(self):
        print("\tSetting up dataloaders...")
        train, val = self.datamanager.get_dataloaders(
            dataset=self.cfg.training_datasets,
            test_size=self.cfg.validation_split,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=self.cfg.shuffle
        )
        return train, val

    def _get_optimizer_and_scheduler(self, params):
        if self.cfg.optimizer_name == "Adam":
            optimizer = optim.Adam(params, lr=self.cfg.learning_rate)
        elif self.cfg.optimizer_name == "SGD":
            optimizer = optim.SGD(params, lr=self.cfg.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer_name}")

        if self.cfg.scheduler_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, **self.cfg.scheduler_params)
        elif self.cfg.scheduler_name == "CosineAnnealingLR":
            scheduler_params = self.cfg.scheduler_params.copy()
            scheduler_params["T_max"] = self.cfg.concept_epochs - self.cfg.warmup_epochs
            scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
        else:
            scheduler = None
        return optimizer, scheduler

    def _freeze_module(self, module):
        for param in module.parameters(): param.requires_grad = False
        module.eval()

    def _unfreeze_module(self, module):
        for param in module.parameters(): param.requires_grad = True
        module.train()

    def _validate_concepts(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels_dict in self.val_loader:
                images, concept_labels = images.to(self.device), labels_dict["concepts"].to(self.device)
                predicted_concepts = self.model.predict_concepts(images)
                loss = self.concept_loss_fn(predicted_concepts, concept_labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def _validate_classifier(self):
        self.model.classifier.eval()
        total_loss, correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for _, labels_dict in self.val_loader:
                concept_labels = labels_dict["concepts"].to(self.device)
                task_labels = labels_dict[self.cfg.label].to(self.device)
                final_output = self.model.classifier(concept_labels)
                loss = self.task_loss_fn(final_output, task_labels)
                total_loss += loss.item()
                _, predicted = torch.max(final_output.data, 1)
                total_samples += task_labels.size(0)
                correct += (predicted == task_labels).sum().item()
        return total_loss / len(self.val_loader), correct / total_samples

    def _train_concepts(self):
        print("\nStep 1: Training Concepts")
        start_time = time.time()
        self._freeze_module(self.model.classifier)
        self._unfreeze_module(self.model.cb_layer)
        if self.cfg.freeze_feature_extractor:
            self._freeze_module(self.model.feature_extractor)
        else:
            self._unfreeze_module(self.model.feature_extractor)

        params_to_train = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer, scheduler = self._get_optimizer_and_scheduler(params_to_train)

        best_val_loss = float('inf')
        patience_counter = 0
        total_warmup_steps = self.cfg.warmup_epochs * len(self.train_loader)

        for epoch in range(self.cfg.concept_epochs):
            self.model.train()
            total_loss = 0
            for i, (images, labels_dict) in enumerate(self.train_loader):
                current_step = epoch * len(self.train_loader) + i
                if current_step < total_warmup_steps:
                    lr_scale = (current_step + 1) / total_warmup_steps
                    new_lr = self.cfg.warmup_lr_init + lr_scale * (self.cfg.learning_rate - self.cfg.warmup_lr_init)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                images, concept_labels = images.to(self.device), labels_dict["concepts"].to(self.device)
                predicted_concepts = self.model.predict_concepts(images)
                loss = self.concept_loss_fn(predicted_concepts, concept_labels)
                optimizer.zero_grad()
                loss.backward()
                if self.cfg.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
                optimizer.step()
                total_loss += loss.item()
                wandb.log(
                    {"train/concept_loss_step": loss.item(),
                     "learning_rate/concepts": optimizer.param_groups[0]['lr'],
                     "global_step": self.global_step}
                )
                self.global_step += 1

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self._validate_concepts()

            wandb.log({
                "train/avg_concept_loss_epoch": avg_train_loss,
                "val/avg_concept_loss_epoch": avg_val_loss,
                "concept_epoch": epoch + 1
            })
            print(f"  Epoch [{epoch+1}/{self.cfg.concept_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if scheduler and epoch >= self.cfg.warmup_epochs:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.cb_layer.state_dict(), self.cfg.output_dir / "best_concept_model.pth")
                print(f"    -> New best model saved with val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"  Stopping early. Validation loss has not improved "
                          f"for {self.cfg.early_stopping_patience} epochs.")
                    break
        print(f"Concept training finished in {time.time() - start_time:.2f} seconds.")

    def _train_classifier(self):
        print("\nStep 2: Training Classifier")
        start_time = time.time()
        self._freeze_module(self.model.feature_extractor)
        self._freeze_module(self.model.cb_layer)
        self._unfreeze_module(self.model.classifier)

        optimizer, scheduler = self._get_optimizer_and_scheduler(self.model.classifier.parameters())

        best_val_accuracy = -1
        patience_counter = 0
        total_warmup_steps = self.cfg.warmup_epochs * len(self.train_loader)

        for epoch in range(self.cfg.classifier_epochs):
            self.model.classifier.train()
            total_loss = 0
            for i, (_, labels_dict) in enumerate(self.train_loader):
                current_step = epoch * len(self.train_loader) + i
                if current_step < total_warmup_steps:
                    lr_scale = (current_step + 1) / total_warmup_steps
                    new_lr = self.cfg.warmup_lr_init + lr_scale * (self.cfg.learning_rate - self.cfg.warmup_lr_init)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                concept_labels = labels_dict["concepts"].to(self.device)
                task_labels = labels_dict[self.cfg.label].to(self.device)
                final_output = self.model.classifier(concept_labels)
                loss = self.task_loss_fn(final_output, task_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log(
                    {"train/task_loss_step": loss.item(),
                     "learning_rate/classifier": optimizer.param_groups[0]['lr'],
                     "global_step": self.global_step}
                )
                self.global_step += 1

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss, val_accuracy = self._validate_classifier()

            wandb.log({
                "train/avg_task_loss_epoch": avg_train_loss,
                "val/avg_task_loss_epoch": avg_val_loss,
                "val/accuracy": val_accuracy,
                "classifier_epoch": epoch + 1
            })
            print(f"  Epoch [{epoch+1}/{self.cfg.classifier_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if scheduler and epoch >= self.cfg.warmup_epochs:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(self.model.classifier.state_dict(), self.cfg.output_dir / "best_classifier.pth")
                print(f"    -> New best classifier saved with val accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"  Stopping early. Validation accuracy has not improved "
                          f"for {self.cfg.early_stopping_patience} epochs.")
                    break
        print(f"Classifier training finished in {time.time() - start_time:.2f} seconds.")

    def train(self, method: str):
        self.global_step = 0
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{self.cfg.dataset}_{method}_{int(time.time())}"

        wandb.init(project=self.cfg.wandb_project, name=run_name, config=vars(self.cfg), entity=self.cfg.wandb_team)

        try:
            if method == "sequential":
                self._train_concepts()
                print("\nLoading best concept model for classifier training...")
                self.model.cb_layer.load_state_dict(torch.load(self.cfg.output_dir / "best_concept_model.pth"))
                self._train_classifier()
            else:
                raise ValueError(f"Unknown training method '{method}'")
        finally:
            wandb.finish()
if __name__ == '__main__':

    cfg = CBMTrainerConfig()
    trainer = CBMTrainer(cfg=cfg)
    trainer.train(method="sequential")