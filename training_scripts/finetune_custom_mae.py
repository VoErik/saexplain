import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import wandb
import numpy as np
import timm
import gc

from src.dataloaders import ImageDataManager, get_transforms


def get_fp_classweights(device, labeltype):
    labels_df = pd.read_csv("../data/fitzpatrick17k/labels.csv")
    class_names = sorted(labels_df[labeltype].unique())
    class_counts = labels_df[labeltype].value_counts().sort_index()
    class_weights = len(labels_df) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
    return class_weights_tensor, class_names

class FineTuningCustomMAE(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        patch_embeddings = self.encoder(x, mask_ratio=0.0)
        image_embedding = torch.mean(patch_embeddings, dim=1)
        logits = self.head(image_embedding)

        return logits

def load_mae_model(ckpt_path="mae-vit-fp-scin-ham.pth", device="cuda", patch_size: int = 14, image_size: int = 224,
                   in_channels: int = 3, dim: int = 768, mlp_ratio: float = 4.0, learned_pos_embed: bool = False):
    args = {
        "image_size": image_size, "patch_size": patch_size, "in_channels": in_channels,
        "dim": dim, "mlp_ratio": mlp_ratio, "learned_pos_embed": learned_pos_embed
    }
    print(f"Loading custom MAE encoder from {ckpt_path}")
    from src.utils.model_utils import load_encoder
    enc = load_encoder(ckpt_path, args)
    enc.to(device)
    print("Custom MAE encoder loaded successfully.")
    return enc

def load_custom_mae_for_finetuning(ckpt_path, num_classes, device="cuda"):
    """
    Loads your custom MAE encoder and wraps it for fine-tuning.
    """
    embedding_dim = 768

    encoder = load_mae_model(ckpt_path=ckpt_path, device=device, dim=embedding_dim)
    full_model = FineTuningCustomMAE(
        encoder=encoder,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )

    full_model.to(device)
    print("Custom MAE model wrapped and ready for fine-tuning.")
    return full_model

def fine_tune_mae(
        model,
        train_loader,
        val_loader,
        class_names,
        device="cuda",
        label_key="label",
        num_epochs: int = 50,
        cw=None,
        encoder_lr: float = 1e-5, # A smaller learning rate for the encoder
        head_lr: float = 1e-3     # A larger learning rate for the new head
):
    print("\n--- Starting MAE Fine-Tuning ---")

    # MODIFIED: Set up optimizer with differential learning rates.
    # We group parameters to apply different LRs.
    param_groups = [
        {"params": model.head.parameters(), "lr": head_lr},
        {"params": model.encoder.parameters(), "lr": encoder_lr} # Reference the wrapped encoder
    ]
    optimizer = optim.AdamW(param_groups)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_val_accuracy = -1.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for images, labels_dict in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            images = images.to(device)
            labels = labels_dict[label_key].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels_dict in val_loader:
                images = images.to(device)
                labels = labels_dict[label_key].to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss_total += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_accuracy = (all_preds == all_labels).mean()
        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}")

        cm = None
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            cm = wandb.plot.confusion_matrix(
                probs=None, y_true=all_labels, preds=all_preds, class_names=class_names
            )

        wandb.log({
            "train/loss": avg_train_loss,
            "val/accuracy": val_accuracy,
            "val/loss": avg_val_loss,
            "val/confusion_matrix": cm,
            "epoch": epoch + 1,
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")
            torch.save(model.state_dict(), "best_mae_finetuned.pth")

    print(f"\nBest Fine-Tuning Accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    labelkeys = ["three_partition_label", "nine_partition_label", "label"]
    seeds = [1, 2, 3, 4, 5]

    for labelkey in labelkeys:
        for seed in seeds:
            gc.collect()
            torch.cuda.empty_cache()

            wandb.init(
                project="custom-mae-finetune-fitzpatrick17k", # New project name
                config={
                    "model_checkpoint": "mae-vit-fp-scin-ham.pth", # Path to your model
                    "batch_size": 32,
                    "encoder_lr": 1e-5,
                    "head_lr": 1e-3,
                    "epochs": 20,
                    "dataset": "fitzpatrick17k",
                }
            )

            print("\n" + "="*50)
            print("Setting up MAE model and data for Fine-Tuning")
            print("="*50)

            mae_dm = ImageDataManager(
                data_root="data",
                initialize="fitzpatrick17k",
                seed=42,
                transform=get_transforms(224)
            )
            train_loader, val_loader = mae_dm.get_dataloaders(
                dataset="fitzpatrick17k",
                batch_size=wandb.config.batch_size,
                shuffle=True, num_workers=8, pin_memory=True, test_size=0.15
            )

            _, class_names = get_fp_classweights(device=device, labeltype=labelkey)

            mae_model = load_custom_mae_for_finetuning(
                ckpt_path=wandb.config.model_checkpoint,
                num_classes=len(class_names),
                device=device
            )

            best = fine_tune_mae(
                model=mae_model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                label_key=labelkey,
                num_epochs=wandb.config.epochs,
                encoder_lr=wandb.config.encoder_lr,
                head_lr=wandb.config.head_lr
            )

            print("Final best validation accuracy:", best)
            wandb.finish()