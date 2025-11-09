import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from pathlib import Path
from tqdm import tqdm
import gc
from typing import List, Any
from dataclasses import dataclass, field, fields


from src.utils.load_backbone import load_backbone
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
    model_architecture: str = "clip"
    model_name: str | None = None
    model_checkpoint: str | None = None
    cache_dir: str = "./cache"
    cls_only: bool = False
    extraction_batch_size: int = 64
    layer_name: list[str] = field(default_factory=list)

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
        self.cfg = cfg

        datasets_str = "-".join(self.cfg.datasets) if isinstance(self.cfg.datasets, list) else self.cfg.datasets
        model_name = Path(self.cfg.model_architecture)

        layer_str = self.cfg.layer_name[0].translate(str.maketrans(".", "-", "[]"))
        
        self.cache_subdir = Path(self.cfg.cache_dir) / datasets_str / model_name / layer_str
        self.train_embeddings_path = self.cache_subdir / "train.safetensors"
        self.eval_embeddings_path = self.cache_subdir / "eval.safetensors"

        if not self._check_if_cache_exists():
            print(f"Cache not found. Extracting embeddings and labels to {self.cache_subdir}...")
            self._extract_embeddings()
        else:
            print(f"Loading cached embeddings and labels from {self.cache_subdir}...")

        self._load_cache()

        self.train_dataset = _CachedDataset(self.train_embeddings, self.train_labels, self.cfg.cls_only)
        self.eval_dataset = _CachedDataset(self.eval_embeddings, self.eval_labels, self.cfg.cls_only)

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
        
        model, _, eval_transform = load_backbone(
            architecture=self.cfg.model_architecture,
            model_name=self.cfg.model_name,
            checkpoint_path=self.cfg.model_checkpoint,
            is_train=False
        )
        model.to(device)
        model.eval()

        train_loader, val_loader = get_dataloaders(
            datasets=self.cfg.datasets, # type: ignore
            data_root=self.cfg.data_root,
            train_transform=eval_transform, # dont apply any data augmentation
            val_transform=eval_transform, 
            batch_size=self.cfg.extraction_batch_size,
            num_workers=4,
        )

        for split, loader in [("train", train_loader), ("eval", val_loader)]:
            print(f"Extracting {split} embeddings and labels...")
            all_embeddings = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Extracting {split}"): # batch is img, label, concepts
                    
                    images, labels, _ = batch
                    images = images.to(device)
                    
                    all_labels.append(labels.cpu())
                    
                    # TODO: make this save the embeddings for an arbitrary amount of layers in parallel
                    features = model.forward_intermediate(images, layer_names=self.cfg.layer_name) # returns Dict[layer_name, ViTOutput]
                    features = features[self.cfg.layer_name[0]]
                    combined = torch.cat([features.cls_embedding.unsqueeze(1), features.patch_embeddings], dim=1)
                    
                    all_embeddings.append(combined.cpu())

            full_embeddings = torch.cat(all_embeddings, dim=0)
            full_labels = torch.cat(all_labels, dim=0)
            
            save_path = self.train_embeddings_path if split == "train" else self.eval_embeddings_path
            print(f"Saving {full_embeddings.shape} embeddings and {full_labels.shape} labels to {save_path}")
            
            save_file(
                {"embeddings": full_embeddings, "labels": full_labels}, 
                save_path
            )
            
            del all_embeddings, full_embeddings, all_labels, full_labels
            gc.collect()

        print("Extraction complete. Deleting model and dataloaders from memory.")
        del model, eval_transform, train_loader, val_loader
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
        self.train_labels = train_data["labels"]
        
        self.eval_embeddings = eval_data["embeddings"]
        self.eval_labels = eval_data["labels"]
        
        print(f"Loaded train embeddings: {self.train_embeddings.shape} and labels: {self.train_labels.shape}")
        print(f"Loaded eval embeddings: {self.eval_embeddings.shape} and labels: {self.eval_labels.shape}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        cache_cfg = CacheConfig.from_dict(config_dict=config_dict)
        res = cls(cache_cfg)
        return res