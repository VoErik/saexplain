import torch
from jaxtyping import Float
from pathlib import Path
import tqdm
from typing import Optional

from src.dataloaders import ImageDataManager
from src.utils.load_backbone import load_encoder



class VisionActivationStore:
    """
    Extracts and stores/caches vision embeddings from an image dataset using a feature extractor.
    Serves batches of these embeddings.
    """

    def __init__(
            self,
            d_in: int,
            dataset_name: str | list = "fitzpatrick17k",
            feature_extractor_model: str = "dinov3_vitl16",
            feature_extractor_path: str = "../dinov3",
            cache_path: str | Path | None = "./embeddings/",
            extraction_batch_size: int = 64,
            shuffle_before_caching: bool = False,
            shuffle_each_epoch: bool = True,
            patches_are_dataset_items: bool = True,
            data_root: str = "data",
            num_workers: int = 8,
            store_batch_size: int = 64,
            device: str | torch.device = "cuda",
            **kwargs
    ):
        self.feature_extractor_model, transform = load_encoder(
            name=feature_extractor_model,
            path=feature_extractor_path,
            device=device,
            **kwargs
        )

        self.dataset_name = dataset_name
        self.dm = ImageDataManager(
            data_root=data_root,
            initialize=self.dataset_name,
            transform=transform
        )
        self.feature_extractor_model.eval()
        self.d_in = d_in
        self.store_batch_size = store_batch_size
        self.device = torch.device(device)
        self.extraction_batch_size = extraction_batch_size
        self.cache_path = cache_path if cache_path else None
        self.shuffle_before_caching = shuffle_before_caching
        self.shuffle_each_epoch = shuffle_each_epoch
        self.num_workers = num_workers

        self.all_pooled_embeddings: Optional[torch.Tensor] = None
        self.all_patch_embeddings: Optional[torch.Tensor] = None
        self._load_or_extract_embeddings()

        self.patches_are_dataset_items = patches_are_dataset_items,
        if self.patches_are_dataset_items:
            num_embeddings = self.all_patch_embeddings.shape[0]
        else:
            num_embeddings = self.all_pooled_embeddings.shape[0]
        self.num_embeddings = num_embeddings
        self.num_batches = (self.num_embeddings + self.store_batch_size - 1) // self.store_batch_size

        self._current_batch_idx = 0
        self._current_shuffled_indices: Optional[torch.Tensor] = None

    def _extract_and_cache_embeddings(self):
        """Runs embedding extraction and caching."""
        print(f"Extracting embeddings using {self.feature_extractor_model.__class__.__name__}...")

        dataloader, _ = self.dm.get_dataloaders(
            dataset=self.dataset_name,
            batch_size=self.extraction_batch_size,
            shuffle=self.shuffle_before_caching,
            num_workers=self.num_workers,
            pin_memory= True if self.device == "cuda" else False,
            test_size=0.0
        )

        extracted_pooled_embeddings_list = []
        extracted_patch_embeddings_list = []
        extracted_metadata_list = []
        self.feature_extractor_model.to(self.device)

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Extracting Embeddings"):
                images, metadata = batch[0], batch[1]
                images = images.to(self.device)

                if "clip" in str(type(self.feature_extractor_model)).lower():
                    self.feature_extractor_model.visual.output_tokens = True
                    pooled_embedding, patch_embeddings = self.feature_extractor_model.visual.forward(images)

                elif "dino" in str(type(self.feature_extractor_model)).lower():
                    with torch.no_grad():
                        x = self.feature_extractor_model.forward_features(images)
                    pooled_embedding = x["x_norm_clstoken"]
                    patch_embeddings = x["x_norm_patchtokens"]

                extracted_pooled_embeddings_list.append(pooled_embedding.cpu())
                extracted_patch_embeddings_list.append(patch_embeddings.cpu())
                extracted_metadata_list.append(metadata)

        self.all_pooled_embeddings = torch.cat(extracted_pooled_embeddings_list, dim=0)
        self.all_patch_embeddings = torch.cat(extracted_patch_embeddings_list, dim=0)

        print(f"Extracted pooled embeddings of dimension {self.all_pooled_embeddings.shape}.")
        print(f"Extracted patch embeddings of dimension {self.all_patch_embeddings.shape}.")

        if self.cache_path:

            print(f"Caching embeddings to {self.cache_path}...")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "pooled_embeddings": self.all_pooled_embeddings,
                    "patch_embeddings": self.all_patch_embeddings,
                    "metadata": extracted_metadata_list,
                },
                self.cache_path
            )
            print("Caching complete.")


    def _load_or_extract_embeddings(self):
        """Loads the embeddings if stored, else extracts them first."""
        if self.cache_path and self.cache_path.exists():
            print(f"Loading cached embeddings from {self.cache_path}...")
            embeddings = torch.load(self.cache_path, map_location='cpu')
            self.all_pooled_embeddings = embeddings["pooled_embeddings"]
            self.all_patch_embeddings = embeddings["patch_embeddings"]
            print(f"Loaded {self.all_pooled_embeddings.shape[0]} embeddings.")
            if self.all_pooled_embeddings.shape[-1] != self.d_in:
                raise ValueError(f"Cached embeddings feature dimension ({self.all_pooled_embeddings.shape[-1]}) "
                                 f"does not match configured d_in ({self.d_in}).")
        else:
            self._extract_and_cache_embeddings()

        if self.all_pooled_embeddings is None:
            raise RuntimeError("Embeddings were not loaded or extracted.")


    def __iter__(self):
        self._current_batch_idx = 0
        if self.shuffle_each_epoch and self.all_pooled_embeddings is not None:
            self._current_shuffled_indices = torch.randperm(self.all_pooled_embeddings.shape[0])
        else:
            self._current_shuffled_indices = None
        return self


    # TODO: Not satisfied with the current implementation.
    def __next__(self) -> Float[torch.Tensor, "batch_size ... d_in"]:
        if self.all_pooled_embeddings is None:
            raise RuntimeError("Accessing store before embeddings are loaded/extracted.")
        if self._current_batch_idx >= self.num_batches:
            raise StopIteration

        start_idx = self._current_batch_idx * self.store_batch_size
        end_idx = min(start_idx + self.store_batch_size, self.num_embeddings)

        if self._current_shuffled_indices is not None:
            indices_for_batch = self._current_shuffled_indices[start_idx:end_idx]
            if self.patches_are_dataset_items:
                batch = self.all_patch_embeddings[indices_for_batch]
            else:
                batch = self.all_pooled_embeddings[indices_for_batch]
        else:
            if self.patches_are_dataset_items:
                batch = self.all_patch_embeddings[start_idx:end_idx]
            else:
                batch = self.all_pooled_embeddings[start_idx:end_idx]

        self._current_batch_idx += 1
        return batch.to(device=self.device)


    def __len__(self) -> int:
        return self.num_batches