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
    A simple Dataset wrapper for cached embeddings.
    Handles the logic for serving CLS token only OR
    flattening all patches into a single dataset.
    """
    def __init__(self, embeddings: torch.Tensor, cls_only: bool):
        self.cls_only = cls_only
        
        if self.cls_only:
            # Shape: [N_samples, Dim]
            self.embeddings = embeddings[:, 0, :] # Take CLS token
            self.is_flat = True
        else:
            # Shape: [N_samples, N_tokens, Dim]
            self.embeddings = embeddings
            self.is_flat = False
            self.n_samples = embeddings.shape[0]
            self.n_tokens = embeddings.shape[1]

    def __len__(self) -> int:
        if self.is_flat:
            return self.embeddings.shape[0] # Total number of CLS tokens
        else:
            return self.n_samples * self.n_tokens 

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.is_flat:
            # Just return the CLS token
            return self.embeddings[idx]
        else:
            sample_idx = idx // self.n_tokens
            token_idx = idx % self.n_tokens
            
            return self.embeddings[sample_idx, token_idx, :]

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
    Caches and serves embeddings from a vision transformer.

    On init, it checks if embeddings for the given model, datasets,
    and layer are already cached. If not, it generates and saves them.
    If yes, it loads them from the cache.

    Provides .train_dataset and .eval_dataset properties that can be
    used directly with a torch.utils.data.DataLoader.
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
            print(f"Cache not found. Extracting embeddings to {self.cache_subdir}...")
            self._extract_embeddings()
        else:
            print(f"Loading cached embeddings from {self.cache_subdir}...")

        self._load_cache()

        self.train_dataset = _CachedDataset(self.train_embeddings, self.cls_only)
        self.eval_dataset = _CachedDataset(self.eval_embeddings, self.cls_only)


    def _check_if_cache_exists(self) -> bool:
        """Checks if both train and eval cache files exist."""
        return (self.train_embeddings_path.is_file() and 
                self.eval_embeddings_path.is_file())


    def _extract_embeddings(self):
        """
        Scenario 1: No cache exists.
        Loads model and dataloaders, runs inference, saves embeddings,
        and cleans up.
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
            print(f"Extracting {split} embeddings...")
            all_embeddings = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Extracting {split}"):
                    if len(batch) > 1:
                        batch = batch[0]
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
            
            save_path = self.train_embeddings_path if split == "train" else self.eval_embeddings_path
            print(f"Saving {full_embeddings.shape} embeddings to {save_path}")
            
            save_file({"embeddings": full_embeddings}, save_path)
            
            del all_embeddings, full_embeddings
            gc.collect()

        print("Extraction complete. Deleting model and dataloaders from memory.")
        del model, transform, train_loader, val_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    
    def _load_cache(self):
        """
        Scenario 2: Cache exists.
        Loads the .safetensors files into memory.
        """
        train_data = load_file(self.train_embeddings_path)
        eval_data = load_file(self.eval_embeddings_path)
        
        self.train_embeddings = train_data["embeddings"]
        self.eval_embeddings = eval_data["embeddings"]
        
        print(f"Loaded train embeddings with shape: {self.train_embeddings.shape}")
        print(f"Loaded eval embeddings with shape: {self.eval_embeddings.shape}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        cache_cfg = CacheConfig.from_dict(config_dict=config_dict)
        res = cls(cache_cfg)
        return res