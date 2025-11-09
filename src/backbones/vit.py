import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, List, Dict, Any

import torch
from safetensors.torch import save_file

T = TypeVar("T")

@dataclass
class ViTOutput:
    """
    Unified output strictly enforcing shapes.
    B = Batch size, N = Number of patches, D = Embedding dimension
    """
    cls_embedding: torch.Tensor  # Shape: (B, D) - CLS token or GAP
    patch_embeddings: torch.Tensor  # Shape: (B, N, D) - Spatial tokens

class ViT(ABC,Generic[T]):
    _model_type: str
    _embedding_dim: int
    _patch_size: int
    _model_name: str
    model: Any

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ViTOutput: ...

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def checkpoint(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def save(self, path: str):
        """
        Saves strictly the backbone weights in safetensors format.
        Compatible with timm.create_model(..., checkpoint_path=path)
        """
        if self.model is None:
            raise RuntimeError("Backbone not initialized, cannot save.")

        if not path.endswith('.safetensors'):
            path += '.safetensors'

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


        save_file(self.model.state_dict(), path)
        print(f"Saved model weights to {path}")

    @property
    def name(self) -> str:
        return f"{self.model_type}/{self.checkpoint}"
    
    @property
    @abstractmethod
    def layers(self) -> list[str]: ...

    def _split_features(self, x: torch.Tensor) -> ViTOutput:
        """Helper to standardize splitting CLS/Registers/Patches."""
        if self.model.num_prefix_tokens == 1:
             return ViTOutput(cls_embedding=x[:, 0], patch_embeddings=x[:, 1:])
        elif self.model.num_prefix_tokens > 1:
             return ViTOutput(cls_embedding=x[:, 0], patch_embeddings=x[:, self.model.num_prefix_tokens:])
        else:
             return ViTOutput(cls_embedding=x.mean(1), patch_embeddings=x)

    def forward_intermediate(self, x: torch.Tensor, layer_names: List[str]) -> Dict[str, ViTOutput]:
        activations = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                # output of a standard timm block is (B, N, D)
                activations[name] = output
            return hook

        try:
            for name, module in self.model.named_modules():
                if name in layer_names:
                    hooks.append(module.register_forward_hook(get_activation(name)))

            self.model(x)

        finally:
            for h in hooks:
                h.remove()

        results = {}
        for name, act in activations.items():
            results[name] = self._split_features(act)

        return results
    
    def freeze_all_except_last_n_blocks(self, n: int):
        """
        Freezes the entire backbone, then unfreezes the last N transformer blocks 
        and the final backbone norm layer.
        """
        print(f"Freezing backbone. Unfreezing last {n} blocks.")
        
        for param in self.model.parameters():
            param.requires_grad = False

        if n > 0:
            if hasattr(self.model, 'norm'):
                 for param in self.model.norm.parameters():
                     param.requires_grad = True

            if hasattr(self.model, 'blocks'):
                total_blocks = len(self.model.blocks)
                start_idx = total_blocks - n
                
                for i in range(start_idx, total_blocks):
                     for param in self.model.blocks[i].parameters():
                         param.requires_grad = True
            else:
                print("Warning: Could not find 'blocks' attribute in backbone. Only standard layers unfrozen.")