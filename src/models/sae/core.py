from pathlib import Path
from typing import Tuple, Any

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from jaxtyping import Float

from src.models.sae.config import SAEConfig, TrainingSAEConfig

class TopK(nn.Module):
    """
    Implements TopK activation, where only the top k active latents are kept.

    Args:
        k (int): Number of latents to keep.
        postact_fn_str (str): Name of the post-activation function.
        **postact_kwargs: Additional keyword arguments to pass to the post-activation function.
    """
    def __init__(self, k: int, postact_fn_str: str = "relu", **postact_fn_kwargs: Any):
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        self.k = k
        self.postact_fn = get_activation_fn(postact_fn_str, **postact_fn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def __repr__(self):
        return f"TopK(k={self.k}, postact_fn={self.postact_fn})"

def get_activation_fn(
        activation_fn_str: str, **kwargs: Any
) -> nn.Module:
    """
    Retrieves an activation function module based on its string name.

    Args:
        activation_fn_str (str): Name of the activation function.
        **kwargs: Additional keyword arguments to pass to the activation function.
    """
    fn_str_lower = activation_fn_str.lower()
    if fn_str_lower == "relu":
        return nn.ReLU(**kwargs)
    elif fn_str_lower == "gelu":
        return nn.GELU(**kwargs)
    elif fn_str_lower == "silu" or fn_str_lower == "swish":
        return nn.SiLU(**kwargs)
    elif fn_str_lower == "tanh":
        return nn.Tanh(**kwargs)
    elif fn_str_lower == "sigmoid":
        return nn.Sigmoid(**kwargs)
    elif fn_str_lower == "tanh-relu": # from sae lens --> first ReLU, then Tanh
        class TanhReLU(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tanh(torch.relu(x))
        return TanhReLU(**kwargs)
    elif fn_str_lower == "topk":
        k_value = kwargs.pop("k", None)
        if k_value is not None:
            postact_fn_str = kwargs.pop("postact_fn_str", "relu")
            return TopK(k=k_value, postact_fn_str=postact_fn_str, **kwargs)
        else:
            raise ValueError("TopK activation function requires a 'k' value in activation_fn_kwargs.")
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn_str}")

class SAE(nn.Module, ABC):
    """
    Abstract Base Class for Sparse Autoencoders.

    This class defines the core interface for an SAE, including encoding
    and decoding functionalities. It handles the configuration and
    activation function setup.
    """
    cfg: SAEConfig
    activation_fn: nn.Module

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.activation_fn = get_activation_fn(
            self.cfg.activation_fn_str, **self.cfg.activation_fn_kwargs
        )


    @abstractmethod
    def encode(self, x: Float[torch.Tensor, "*batch d_in"]) -> Float[torch.Tensor, "*batch d_sae"]:
        """
        Encodes the input tensor x into sparse feature activations.
        Args:
            x: Input tensor of shape (..., d_in).
        Returns:
            Tensor of feature activations of shape (..., d_sae).
        """
        pass

    @abstractmethod
    def encode_with_hidden_pre(
            self, x: Float[torch.Tensor, "*batch d_in"]
    ) -> Tuple[Float[torch.Tensor, "*batch d_sae"], Float[torch.Tensor, "*batch d_sae"]]:
        """
        Encodes input, returning both feature activations and pre-activations.
        Returns: (feature_activations, hidden_pre_activations)
        """
        pass

    @abstractmethod
    def decode(self, features: Float[torch.Tensor, "*batch d_sae"]) -> Float[torch.Tensor, "*batch d_in"]:
        """
        Decodes sparse feature activations back into the input space.
        Args:
            features: Tensor of feature activations of shape (..., d_sae).
        Returns:
            Reconstructed input tensor of shape (..., d_in).
        """
        pass

    def forward(self, x: Float[torch.Tensor, "*batch d_in"]) -> Float[torch.Tensor, "*batch d_in"]:
        """
        Full forward pass: encode the input, then decode the features.
        Args:
            x: Input tensor of shape (..., d_in).
        Returns:
            Reconstructed input tensor of shape (..., d_in).
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction

    @property
    def device(self) -> torch.device:
        """Returns the device of the model's parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the model's parameters."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32 # Default to float32

    def get_config(self) -> dict:
        return self.cfg.to_dict()

    def get_name(self) -> str:
        name_parts = [
            "sae",
            self.cfg.architecture,
            f"d_in_{self.cfg.d_in}",
            f"d_sae_{self.cfg.d_sae}",
            f"act_{self.cfg.activation_fn_str}"
        ]
        if (hasattr(self.cfg, 'activation_fn_kwargs') and ('k' in self.cfg.activation_fn_kwargs)
                and (self.cfg.architecture == "topk")):
            name_parts.append(f"k{self.cfg.activation_fn_kwargs['k']}")
        if hasattr(self.cfg, 'embedding_source_name') and self.cfg.embedding_source_name:
            name_parts.insert(1, self.cfg.embedding_source_name)
        if isinstance(self.cfg, TrainingSAEConfig):
            if self.cfg.architecture != "topk": name_parts.append(f"l1_{self.cfg.l1_coefficient}")
            if self.cfg.architecture == "topk": name_parts.append(f"auxcoeff_{self.cfg.topk_aux_loss_coefficient}")
        return "_".join(name_parts)

    def save_model(self, dir_path: str | Path):
        """Saves the model and its configuration to the specified directory."""
        from src.models.sae.utils import save_sae
        save_sae(self, dir_path)

    @classmethod
    def load_model(
            cls,
            dir_path: str | Path,
            device: str | torch.device | None = None
    ) -> 'SAE':
        """
        Loads a model and its configuration from the specified directory.

        Args:
            dir_path: Path to the directory to load the model from.
            device: Device to load the model to.
        """
        from src.models.sae.utils import load_sae
        return load_sae(dir_path, device=device)