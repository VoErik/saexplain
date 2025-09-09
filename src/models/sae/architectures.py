from typing import Tuple

import torch
import torch.nn as nn
from einops import einops
from jaxtyping import Float

from src.models.sae.config import SAEConfig, TrainingSAEConfig
from src.models.sae.core import SAE

###################################################################################
################################ STANDARD #########################################
###################################################################################

class StandardSAE(SAE):
    """
    A basic Sparse Autoencoder implementation.

    This SAE consists of an encoder and a decoder, each being a single
    linear layer followed by biases. The encoder output passes through
    an ReLU activation function.
    """
    W_enc: nn.Parameter
    b_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg: SAEConfig | TrainingSAEConfig):
        super().__init__(cfg)

        # Encoder params
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.empty(cfg.d_sae))

        # Decoder params
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.b_dec = nn.Parameter(torch.empty(cfg.d_in))

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights and biases of the SAE based on activation function.
        """
        if self.cfg.activation_fn_str in ["relu", "topk", "jumprelu"]:
            nonlinearity = "relu"
        else:
            nonlinearity = "linear"

        if self.cfg.activation_fn_str in ["tanh", "sigmoid"]:
            nn.init.xavier_uniform_(self.W_enc)
            nn.init.xavier_uniform_(self.W_dec)
        else:
            nn.init.kaiming_uniform_(self.W_enc, a=0, mode='fan_in', nonlinearity=nonlinearity)
            nn.init.kaiming_uniform_(self.W_dec, a=0, mode='fan_in', nonlinearity=nonlinearity)

        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def encode(
            self, x: Float[torch.Tensor, "*batch d_in"]
    ) -> Float[torch.Tensor, "*batch d_sae"]:
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def encode_with_hidden_pre(
            self,
            x: Float[torch.Tensor, "*batch d_in"]
    ) -> Tuple[Float[torch.Tensor, "*batch d_sae"], Float[torch.Tensor, "*batch d_sae"]]:
        """Returns both pre-activation function features and post activation function features."""
        x_processed = x.to(dtype=self.dtype, device=self.device)
        if self.cfg.apply_b_dec_to_input:
            input_centered = x_processed - self.b_dec
        else:
            input_centered = x_processed

        hidden_pre_activations = einops.einsum(
            input_centered, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_enc
        feature_activations = self.activation_fn(hidden_pre_activations)

        return feature_activations, hidden_pre_activations

    def decode(
            self,
            features: Float[torch.Tensor, "*batch d_sae"]
    ) -> Float[torch.Tensor, "*batch d_in"]:
        """
        Decodes the feature activations.
        """
        features_processed = features.to(dtype=self.dtype, device=self.device)
        reconstruction = einops.einsum(
            features_processed, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec

        return reconstruction

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Removes the component of the gradient parallel to the decoder directions.
        This is used to ensure that the decoder weights remain on the unit sphere.
        """
        if self.W_dec.grad is None:
            return

        # Project the gradient onto the decoder vectors
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )

        # Subtract the parallel component from the gradient
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

###################################################################################
##################################### TOPK ########################################
###################################################################################

class TopKSAE(StandardSAE):
    """
    Sparse Autoencoder using Top-K activation.
    Inherits most behavior from StandardSAE, but its activation function
    is specifically a TopK module.
    """
    def __init__(self, cfg: SAEConfig | TrainingSAEConfig):
        super().__init__(cfg)
        if cfg.architecture not in ["topk", "batchtopk"]:
            raise ValueError("TopKSAE class instantiated with non-'topk' architecture in config.")
        if cfg.activation_fn_str.lower() != "topk":
            print(f"Warning: TopKSAE initialized with activation_fn_str='{cfg.activation_fn_str}'. "
                  "Ensure this is intended and 'k' is provided in activation_fn_kwargs.")
        if cfg.architecture == "batchtopk":
            cfg.topk_mode = "batch"
            print(f"Using {cfg.topk_mode}.")
        else:
            cfg.topk_mode = "instance"


###################################################################################
##################################### GATED #######################################
###################################################################################

class GatedSAE(StandardSAE):
    """
    DeepMind-style Gated Sparse Autoencoder.

    Inherits from StandardSAE but overrides the encoder to use:
      - Gating path (binary mask from shared W_enc + b_gate)
      - Magnitude path (rescaled W_enc with r_mag, plus b_mag)
    """

    b_gate: nn.Parameter
    b_mag: nn.Parameter
    r_mag: nn.Parameter

    def __init__(self, cfg: SAEConfig | TrainingSAEConfig):
        super().__init__(cfg)

        # Gated SAEs don't use b_enc
        self.b_enc = None

        # Additional gated parameters
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_mag = nn.Parameter(torch.zeros(cfg.d_sae))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_sae))  # log scaling

    def encode_with_hidden_pre(
            self,
            x: Float[torch.Tensor, "*batch d_in"],
    ) -> Tuple[Float[torch.Tensor, "*batch d_sae"], Float[torch.Tensor, "*batch d_sae"]]:
        """Returns both post-gating activations and the magnitude pre-activations."""

        x_processed = x.to(dtype=self.dtype, device=self.device)
        if self.cfg.apply_b_dec_to_input:
            input_centered = x_processed - self.b_dec
        else:
            input_centered = x_processed

        # Gating path
        gating_pre = einops.einsum(
            input_centered, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_gate
        active = (gating_pre > 0).to(self.dtype)

        # Magnitude path
        magnitude_pre = einops.einsum(
            input_centered,
            self.W_enc * self.r_mag.exp(),
            "... d_in, d_in d_sae -> ... d_sae",
            ) + self.b_mag
        feature_magnitudes = self.activation_fn(magnitude_pre)

        # Final gated activations
        feature_activations = active * feature_magnitudes

        return feature_activations, magnitude_pre

