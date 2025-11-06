import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from pathlib import Path
from tqdm import tqdm
import gc
from typing import List, Any
from dataclasses import dataclass, field, fields


from src.backbones.utils import load_backbone
from sae.utils.misc import filter_valid_dataclass_fields
from src.data import get_dataloaders


class _CachedDataset(Dataset):
    """
    A simple Dataset wrapper for cached embeddings and labels.
    Handles the logic for serving CLS token only OR
    flattening all patches into a single dataset.
    """
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor, cls_only: bool):
        self.cls_only = cls_only
        self.labels = labels # Shape: [N_samples]
        
        if self.cls_only:
            # Shape: [N_samples, Dim]
            self.embeddings = embeddings[:, 0, :] # Take CLS token
            self.is_flat = True
            # Sanity check
            assert self.embeddings.shape[0] == self.labels.shape[0], "Embeddings and labels have different sample sizes!"
        else:
            # Shape: [N_samples, N_tokens, Dim]
            self.embeddings = embeddings
            self.is_flat = False
            self.n_samples = embeddings.shape[0]
            self.n_tokens = embeddings.shape[1]
            # Sanity check
            assert self.n_samples == self.labels.shape[0], "Embeddings and labels have different sample sizes!"

    def __len__(self) -> int:
        if self.is_flat:
            return self.embeddings.shape[0] # Total number of CLS tokens
        else:
            return self.n_samples * self.n_tokens 

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_flat:
            # Just return the CLS token and its label
            embedding = self.embeddings[idx]
            label = self.labels[idx]
            return embedding, label
        else:
            sample_idx = idx // self.n_tokens
            token_idx = idx % self.n_tokens
            
            embedding = self.embeddings[sample_idx, token_idx, :]
            # The label is the same for all tokens from the same sample
            label = self.labels[sample_idx] 
            
            return embedding, label

@dataclass
class CacheConfig:
    datasets: list[str] = field(default_factory=list)
    data_root: str = "../data"
    model_path: str = "openai/clip-vit-base-patch32"
    cache_dir: str = "./cache"
    cls_only: bool = False
    extraction_batch_size: int = 64
    layer_index: int = -1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res
    

    def to_dict(self) -> dict[str, Any]:
        res = {field.name: getattr(self, field.name) for field in fields(self)}
        return res

class EmbeddingCache:
    """
    Caches and serves embeddings AND labels from a vision transformer.
    ...
    """
    def __init__(
            self,
            cfg: CacheConfig
    ):
        self.datasets = cfg.datasets
        self.data_root = cfg.data_root
        self.model_path = cfg.model_path
        self.save_dir = Path(cfg.cache_dir)
        self.cls_only = cfg.cls_only
        self.extraction_batch_size = cfg.extraction_batch_size
        self.layer_index = cfg.layer_index

        datasets_str = "-".join(self.datasets) if isinstance(self.datasets, list) else self.datasets
        model_name = Path(self.model_path).name 
        layer_str = f"layer_{self.layer_index}"
        
        self.cache_subdir = self.save_dir / datasets_str / model_name / layer_str
        self.train_embeddings_path = self.cache_subdir / "train.safetensors"
        self.eval_embeddings_path = self.cache_subdir / "eval.safetensors"

        if not self._check_if_cache_exists():
            print(f"Cache not found. Extracting embeddings and labels to {self.cache_subdir}...")
            self._extract_embeddings()
        else:
            print(f"Loading cached embeddings and labels from {self.cache_subdir}...")

        self._load_cache()

        # --- MODIFIED ---
        # Pass the labels to the dataset
        self.train_dataset = _CachedDataset(self.train_embeddings, self.train_labels, self.cls_only)
        self.eval_dataset = _CachedDataset(self.eval_embeddings, self.eval_labels, self.cls_only)

    def _check_if_cache_exists(self) -> bool:
        """Checks if both train and eval cache files exist."""
        return (self.train_embeddings_path.is_file() and 
                self.eval_embeddings_path.is_file())

    def _extract_embeddings(self):
        """
        ...
        Loads model and dataloaders, runs inference, saves embeddings AND labels,
        ...
        """
        self.cache_subdir.mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for extraction.")
        
        model, transform = load_backbone(self.model_path, is_train=False)
        model.to(device)
        model.eval()

        train_loader, val_loader = get_dataloaders(
            datasets=self.datasets, # type: ignore
            data_root=self.data_root, 
            transform=transform, 
            batch_size=self.extraction_batch_size,
            num_workers=4,
        )

        for split, loader in [("train", train_loader), ("eval", val_loader)]:
            print(f"Extracting {split} embeddings and labels...")
            all_embeddings = []
            all_labels = [] # <-- NEW
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Extracting {split}"): # batch is img, label, concepts
                    
                    # --- NEW: Extract Labels ---
                    # This logic handles two common dataloader structures:
                    # 1. A tuple: (data_dict, labels)
                    # 2. A dict: {"pixel_values": ..., "labels": ...}
                    labels_batch = None
                    if isinstance(batch, (list, tuple)) and len(batch) > 1:
                        labels_batch = batch[1] # Assume (data, labels)
                        batch = batch[0]        # Re-assign batch to be the data dict
                    elif isinstance(batch, dict) and "labels" in batch:
                        labels_batch = batch["labels"]
                    
                    if labels_batch is None:
                        raise ValueError("Could not find 'labels' in the batch. "
                                         "Ensure your dataloader yields (data, labels) "
                                         "or a dict with a 'labels' key.")
                    
                    all_labels.append(labels_batch.cpu())
                    # --- END NEW ---

                    images = batch["pixel_values"].to(device)
                    
                    features = model(
                        pixel_values=images,
                        return_patch_embeddings=True,
                        layer_index=self.layer_index
                    )
                    if len(features) > 1:
                        features = features[0]
                    
                    all_embeddings.append(features.cpu())

            full_embeddings = torch.cat(all_embeddings, dim=0)
            full_labels = torch.cat(all_labels, dim=0) # <-- NEW
            
            save_path = self.train_embeddings_path if split == "train" else self.eval_embeddings_path
            print(f"Saving {full_embeddings.shape} embeddings and {full_labels.shape} labels to {save_path}")
            
            # --- MODIFIED: Save both tensors ---
            save_file(
                {"embeddings": full_embeddings, "labels": full_labels}, 
                save_path
            )
            
            del all_embeddings, full_embeddings, all_labels, full_labels # <-- MODIFIED
            gc.collect()

        print("Extraction complete. Deleting model and dataloaders from memory.")
        del model, transform, train_loader, val_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    
    def _load_cache(self):
        """
        ...
        Loads the .safetensors files (embeddings and labels) into memory.
        """
        train_data = load_file(self.train_embeddings_path)
        eval_data = load_file(self.eval_embeddings_path)
        
        self.train_embeddings = train_data["embeddings"]
        self.train_labels = train_data["labels"] # <-- NEW
        
        self.eval_embeddings = eval_data["embeddings"]
        self.eval_labels = eval_data["labels"] # <-- NEW
        
        print(f"Loaded train embeddings: {self.train_embeddings.shape} and labels: {self.train_labels.shape}")
        print(f"Loaded eval embeddings: {self.eval_embeddings.shape} and labels: {self.eval_labels.shape}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        cache_cfg = CacheConfig.from_dict(config_dict=config_dict)
        res = cls(cache_cfg)
        return res