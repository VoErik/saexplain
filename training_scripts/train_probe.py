import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.utils.model_utils import load_encoder
from src.dataloaders import ImageDataManager, get_transforms

from pathlib import Path
import torch
import pandas as pd

def get_skincon_weights(device, labeltype):
    path = "../data/fitzpatrick17k/labels.csv"
    skincon = "data/fitzpatrick17k/skincon-annotations.csv"
    images_dir = Path("../data/fitzpatrick17k/images")
    labels_df = pd.read_csv(path)
    skincon_df = pd.read_csv(skincon)

    excluded_cols = ['ImageID', '<anonymous>', 'Do not consider this image']
    concept_columns = sorted([
        col for col in skincon_df.columns if col not in excluded_cols
    ])

    skincon_df[concept_columns] = skincon_df[concept_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    skincon_df = skincon_df[skincon_df[concept_columns].sum(axis=1) > 0]
    skincon_prepared = skincon_df[concept_columns + ['ImageID']].set_index('ImageID')

    labels_df["ImageID"] = labels_df["md5hash"].astype(str) + ".jpg"
    labels_df["image_path"] = labels_df["ImageID"].apply(lambda x: images_dir / x)
    master_df = labels_df[labels_df["image_path"].apply(lambda x: x.exists())]

    metadata = master_df.set_index('ImageID').join(skincon_prepared, how='inner').reset_index()

    metadata[concept_columns] = metadata[concept_columns].fillna(0)

    class_counts = metadata[labeltype].value_counts().sort_index()
    class_weights = len(metadata) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
    return class_weights_tensor

ckpt_path = "mae-vit-fp-scin-ham.pth"
args = {
    "image_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "dim": 768,
    "mlp_ratio": 4.0,
    "learned_pos_embed": False
}

labeltype = "nine_partition_label"
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 8
PIN_MEMORY = True
TEST_SIZE = 0.15
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100 # You can adjust this
OUTPUT_DIR = "./out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

enc = load_encoder(ckpt_path, args)
enc.to(device)
enc.eval()

dm = ImageDataManager(
    data_root="data",
    initialize="fitzpatrick17k",
    seed=42,
    transform=get_transforms(224)
)

train_loader, val_loader = dm.get_dataloaders(
    dataset="fitzpatrick17k",
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    test_size=TEST_SIZE
)

num_classes = 9
embedding_dim = args["dim"]

linear_classifier = nn.Linear(embedding_dim, 512)
linear_classifier.to(device)


cw = get_skincon_weights(device, labeltype)
loss_fn = nn.CrossEntropyLoss(weight=cw).to(device)
optimizer = optim.Adam(linear_classifier.parameters(), lr=LEARNING_RATE)

best_val_accuracy = -1.0

for epoch in range(NUM_EPOCHS):
    linear_classifier.train()
    total_train_loss = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for images, labels_dict in train_loop:
        images = images.to(device)
        labels = labels_dict[labeltype].to(device)

        with torch.no_grad():
            embeddings = enc.forward(images, mask_ratio=0.0)
            pooled_embeddings = torch.mean(embeddings, dim=1)

        outputs = linear_classifier(pooled_embeddings)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    linear_classifier.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        for images, labels_dict in val_loop:
            images = images.to(device)
            labels = labels_dict[labeltype].to(device)

            with torch.no_grad():
                embeddings = enc.forward(images, mask_ratio=0.0)
            pooled_embeddings = torch.mean(embeddings, dim=1)

            outputs = linear_classifier(pooled_embeddings)

            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_predictions / total_samples

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        save_path = os.path.join(OUTPUT_DIR, "best_linear_classifier.pth")
        torch.save(linear_classifier.state_dict(), save_path)
        print(f"  -> New best model saved with accuracy: {best_val_accuracy:.4f}")

print("\nTraining finished.")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")