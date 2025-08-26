import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import open_clip
import pandas as pd
import wandb
import numpy as np
import timm
import gc


from src.dataloaders import ImageDataManager, get_transforms
from src.utils.model_utils import load_encoder


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, model, dataloader, device, label_key="label", cache_file=None):
        self.embeddings = []
        self.labels = []

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            cache = torch.load(cache_file)
            self.embeddings = cache["embeddings"]
            self.labels = cache["labels"]
        else:
            print("Computing MAE embeddings...")
            model.eval()
            with torch.no_grad():
                for images, labels_dict in tqdm(dataloader, desc="Precomputing embeddings"):
                    images = images.to(device)
                    labels = labels_dict[label_key].to(device)

                    embeddings = mae_model.forward(images)
                    if embeddings.ndim != 2:
                        feats = torch.mean(embeddings, dim=1)
                    else:
                        feats = embeddings

                    self.embeddings.append(feats.detach().cpu())
                    self.labels.append(labels.cpu())

            self.embeddings = torch.cat(self.embeddings)
            self.labels = torch.cat(self.labels)

            if cache_file:
                torch.save({"embeddings": self.embeddings, "labels": self.labels}, cache_file)
                print(f"Saved embeddings to {cache_file}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def get_fp_classweights(device, labeltype):
    labels_df = pd.read_csv("../data/fitzpatrick17k/labels.csv")
    class_names = sorted(labels_df[labeltype].unique())
    class_counts = labels_df[labeltype].value_counts().sort_index()
    class_weights = len(labels_df) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
    return class_weights_tensor, class_names

def load_mae_model(ckpt_path="mae-vit-fp-scin-ham.pth", device="cuda", patch_size: int = 14, image_size: int = 224,
                   in_channels: int = 3, dim: int = 768, mlp_ratio: float = 4.0, learned_pos_embed: bool = False):
    """
    Loads a pretrained MAE model.
    """
    args = {
        "image_size": image_size,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "dim": dim,
        "mlp_ratio": mlp_ratio,
        "learned_pos_embed": learned_pos_embed
    }
    print(f"Loading MAE model")
    enc = load_encoder(ckpt_path, args)
    enc.to(device)
    enc.to(device)
    print("MAE model loaded successfully.")
    return enc

def load_pretrained_mae_model(architecture: str = "vit_base_patch16_224", freeze: bool = True, device: str = "cuda"):
    # Load a MAE-pretrained ViT
    print("Loading pretrained MAE")
    mae_model = timm.create_model(architecture, pretrained=True, features_only=False)

    mae_model.eval()
    mae_model.to(device)

    if freeze:
        for param in mae_model.parameters():
            param.requires_grad = False
    print("MAE loaded successfully")
    return mae_model

def train_mae_linear_probe(
        mae_model,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        device="cuda",
        label_key="label",
        num_epochs: int = 50,
        use_cached_embeddings: bool = False,
        cw=None,
        embedding_dim: int = 768
):
    print("\n--- Training Linear Probe ---")

    if use_cached_embeddings:
        sample_feats, _ = next(iter(train_loader))
        embedding_dim = sample_feats.shape[-1]
    else:
        embedding_dim = embedding_dim
        mae_model.eval()

    linear_classifier = nn.Linear(embedding_dim, num_classes).to(device)
    optimizer = optim.Adam(linear_classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_val_accuracy = -1.0

    for epoch in range(num_epochs):
        linear_classifier.train()
        total_train_loss = 0.0

        if use_cached_embeddings:
            for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
                feats, labels = feats.to(device), labels.to(device)
                outputs = linear_classifier(feats)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
        else:
            for images, labels_dict in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
                images = images.to(device)
                labels = labels_dict[label_key].to(device)

                with torch.no_grad():
                    embeddings = mae_model.forward(images)
                    if embeddings.ndim != 2:
                        feats = torch.mean(embeddings, dim=1)
                    else:
                        feats = embeddings

                outputs = linear_classifier(feats)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        linear_classifier.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0

        if use_cached_embeddings:
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    outputs = linear_classifier(feats)
                    loss = loss_fn(outputs, labels)
                    val_loss_total += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else:
            with torch.no_grad():
                for images, labels_dict in val_loader:
                    images = images.to(device)
                    labels = labels_dict[label_key].to(device)

                    with torch.no_grad():
                        embeddings = mae_model.forward(images)
                        if embeddings.ndim != 2:
                            feats = torch.mean(embeddings, dim=1)
                        else:
                            feats = embeddings
                    outputs = linear_classifier(feats)
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

        per_class_acc = {}
        for i, cname in enumerate(class_names):
            mask = (all_labels == i)
            per_class_acc[cname] = float((all_preds[mask] == all_labels[mask]).mean()) if mask.sum() > 0 else float("nan")

        cm = None
        if epoch % 5 == 0 or epoch == num_epochs:
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )

        wandb.log({
            "train/loss": avg_train_loss,
            "val/accuracy": val_accuracy,
            "val/loss": avg_val_loss,
            "val/confusion_matrix": cm,
            "val/per_class_accuracy": per_class_acc,
            "epoch": epoch + 1,
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")

    print(f"\nBest Linear Probe Accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    PRETRAINED = True
    labelkeys = ["three_partition_label", "nine_partition_label", "label"]
    seeds = [1,2,3,4,5]
    gc.collect()
    torch.cuda.empty_cache()
    for labelkey in labelkeys:
        for seed in seeds:
            gc.collect()
            torch.cuda.empty_cache()
            wandb.init(
                project="mae-fitzpatrick17k",
                config={
                    "model": "MAE_Patch16_Dim768",
                    "pretrained": "imagenet1k",
                    "batch_size": 64,
                    "lr": 1e-3,
                    "epochs": 50,
                    "dataset": "fitzpatrick17k",
                    "use_cached_embeddings": True
                }
            )

            print("\n" + "="*50)
            print("Setting up MAE model and data")
            print("="*50)

            if PRETRAINED:
                mae_model = load_pretrained_mae_model(device=device)
            else:
                mae_model = load_mae_model(device=device)

            mae_dm = ImageDataManager(
                data_root="data",
                initialize="fitzpatrick17k",
                seed=42,
                transform=get_transforms(224)
            )

            mae_train_loader, mae_val_loader = mae_dm.get_dataloaders(
                dataset="fitzpatrick17k",
                batch_size=64,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                test_size=0.15
            )

            use_cached_embeddings = wandb.config.use_cached_embeddings

            if use_cached_embeddings:
                train_embed_dataset = EmbeddingDataset(
                    model=mae_model,
                    dataloader=mae_train_loader,
                    device=device,
                    label_key=labelkey,
                    cache_file=f"{labelkey}_train_embeddings.pt"
                )
                val_embed_dataset = EmbeddingDataset(
                    model=mae_model,
                    dataloader=mae_val_loader,
                    device=device,
                    label_key=labelkey,
                    cache_file=f"{labelkey}_val_embeddings.pt"
                )
                train_loader = DataLoader(train_embed_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_embed_dataset, batch_size=64, shuffle=False)
            else:
                train_loader, val_loader = mae_train_loader, mae_val_loader

            _, class_names = get_fp_classweights(device=device, labeltype=labelkey)


            best = train_mae_linear_probe(
                mae_model=mae_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=len(class_names),
                class_names=class_names,
                device=device,
                label_key=labelkey,
                num_epochs=50,
                use_cached_embeddings=use_cached_embeddings,
            )

            print("Final best validation accuracy:", best)
            wandb.finish()