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
from pathlib import Path

from src.dataloaders import ImageDataManager, get_transforms


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
            print("Computing DINO embeddings...")
            model.eval()
            with torch.no_grad():
                for images, labels_dict in tqdm(dataloader, desc="Precomputing embeddings"):
                    images = images.to(device)
                    labels = labels_dict[label_key].to(device)

                    embeddings = model.forward(images)
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

def load_dinov2_model(model_name="dinov2_vitl14", device="cuda"):
    """
    Loads a pretrained DINOv2 model and its image preprocessor.
    """
    print(f"Loading DINOv2 model: {model_name}")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("DINOv2 model loaded successfully.")
    return model, preprocess

def get_dinov3_weights(model_name):
    WEIGHTS_DICT = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vit7b16": None
    }
    return WEIGHTS_DICT[model_name]

def load_dinov3_model(model_name="dinov3_vith16plus", device="cuda"):
    """
    Loads a pretrained DINOv3 model and its image preprocessor.
    """
    print(f"Loading DINOv2 model: {model_name}")
    REPO_DIR = Path("../dinov3")
    WEIGHT_PATH = REPO_DIR / "dinov3_weights"/ get_dinov3_weights(model_name)
    model = torch.hub.load(str(REPO_DIR), model_name, source='local', weights=str(WEIGHT_PATH))
    model = model.to(device)

    from torchvision import transforms
    def make_transform(resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([to_tensor, resize, normalize])

    preprocess = make_transform()

    print("DINOv3 model loaded successfully.")
    return model, preprocess

def train_linear_probe(
        model,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        device="cuda",
        label_key="label",
        num_epochs: int = 50,
        use_cached_embeddings: bool = False,
        cw=None
):
    print("\n--- Training Linear Probe ---")

    if use_cached_embeddings:
        sample_feats, _ = next(iter(train_loader))
        embedding_dim = sample_feats.shape[-1]
    else:
        embedding_dim = model.embed_dim
        model.eval()

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
                    feats = model(images)

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

                    feats = model(images)
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
        if epoch % 5 == 0 or epoch == num_epochs - 1:
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
                project="dinov3-fitzpatrick17k",
                config={
                    "model": "dinov3_vith16plus",
                    "batch_size": 64,
                    "lr": 1e-3,
                    "epochs": 50,
                    "dataset": "fitzpatrick17k",
                    "use_cached_embeddings": True
                }
            )

            print("\n" + "="*50)
            print("Setting up DINOv2 model and data")
            print("="*50)

            dino_model, dino_preprocess = load_dinov3_model(
                model_name=wandb.config.model,
                device=device
            )

            dino_dm = ImageDataManager(
                data_root="data",
                initialize="fitzpatrick17k",
                seed=seed,
                transform=dino_preprocess
            )

            mae_train_loader, mae_val_loader = dino_dm.get_dataloaders(
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
                    model=dino_model,
                    dataloader=mae_train_loader,
                    device=device,
                    label_key=labelkey,
                    cache_file=f"dinov2_{labelkey}_train_embeddings.pt"
                )
                val_embed_dataset = EmbeddingDataset(
                    model=dino_model,
                    dataloader=mae_val_loader,
                    device=device,
                    label_key=labelkey,
                    cache_file=f"dinov2_{labelkey}_val_embeddings.pt"
                )
                train_loader = DataLoader(train_embed_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_embed_dataset, batch_size=64, shuffle=False)
            else:
                train_loader, val_loader = mae_train_loader, mae_val_loader

            _, class_names = get_fp_classweights(device=device, labeltype=labelkey)


            best = train_linear_probe(
                model=dino_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=len(class_names),
                class_names=class_names,
                device=device,
                label_key=labelkey,
                num_epochs=wandb.config.epochs,
                use_cached_embeddings=use_cached_embeddings,
            )

            print("Final best validation accuracy:", best)
            wandb.finish()