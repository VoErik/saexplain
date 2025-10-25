from dataclasses import dataclass
from typing_extensions import override

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from numpy.typing import NDArray
from src.sae.base import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAEConfig,
    TrainingSAE,
    TrainStepInput,
)

@dataclass
class ReLUSAEConfig(SAEConfig):
    """
    Configuration class for a ReLUSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "relu"

@dataclass
class ReLUSAETrainingConfig(TrainingSAEConfig):
    """
    Configuration class for training a ReLUTrainingSAE.
    """

    l1_coefficient: float = 1.0
    lp_norm: float = 1.0
    l1_warm_up_steps: int = 0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "relu"

class ReLUSAE(SAE):

    b_enc: nn.Parameter

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.initialize_weights()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_standard(self)
    
    def encode(self, x):
        sae_in = self.process_sae_in(x)
        features = einops.einsum(
            sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_enc
        return features
    
    def decode(self, features):
        sae_out = einops.einsum(
            features, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec
        return sae_out
    

class ReLUTrainingSAE(TrainingSAE):

    b_enc: nn.Parameter

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_standard(self)

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l1": TrainCoefficientConfig(
                value=self.cfg.l1_coefficient,
                warm_up_steps=self.cfg.l1_warm_up_steps,
            ),
        }

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        sae_in = self.process_sae_in(x)
        hidden_pre_activations = einops.einsum(
            sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_enc
        feature_acts = self.activation_fn(hidden_pre_activations)
        return feature_acts, hidden_pre_activations

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # The "standard" auxiliary loss is a sparsity penalty on the feature activations
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)

        # Compute the p-norm (set by cfg.lp_norm) over the feature dimension
        sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        l1_loss = (step_input.coefficients["l1"] * sparsity).mean()

        return {"l1_loss": l1_loss}

    def log_histograms(self) -> dict[str, NDArray[np.generic]]:
        """Log histograms of the weights and biases."""
        b_e_dist = self.b_enc.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_e": b_e_dist,
        }

def _init_weights_standard(
    sae,
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )

if __name__ == "__main__":
    base_cfg = ReLUSAEConfig()
    train_cfg = ReLUSAETrainingConfig()

    relu_sae = ReLUSAE(cfg=base_cfg)
    train_relu_sae = ReLUTrainingSAE(cfg=train_cfg)

    dummy_input_cls = torch.randn(2, 768)
    dummy_input_patches = torch.randn(2, 50, 768)

    out_cls_base = relu_sae(dummy_input_cls)
    out_patches_base = relu_sae(dummy_input_patches)

    out_cls_train = train_relu_sae(dummy_input_cls)
    out_patches_train = train_relu_sae(dummy_input_patches)

    col_width = 50

    print(f"{f'Expected shape CLS = {dummy_input_cls.shape}': <{col_width}} || Actual shape base ReLUSAE = {out_cls_base.shape}")
    print(f"{f'Expected shape Patches = {dummy_input_patches.shape}': <{col_width}} || Actual shape base ReLUSAE = {out_patches_base.shape}") 
    print(f"{f'Expected shape CLS = {dummy_input_cls.shape}': <{col_width}} || Actual shape train ReLUSAE = {out_cls_train.shape}") 
    print(f"{f'Expected shape Patches = {dummy_input_patches.shape}': <{col_width}} || Actual shape train ReLUSAE = {out_patches_train.shape}")