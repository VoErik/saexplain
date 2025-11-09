from typing import Callable, Any
from typing_extensions import override

import einops
import torch
import torch.nn as nn

from jaxtyping import Float

from sae.core import (
    SAE,
    TrainingSAE,
    SAEConfig,
    TrainingSAEConfig,
    TrainStepInput,
    TrainCoefficientConfig
)


class TopKActivation(nn.Module):
    """
    TopK activation function.
    
    Identifies the top k largest values in the tensor and applies a ReLU to them. 
    Then zeroes out every other value.

    Args:
        k (int): How many active neurons to keep. Defaults to 20.
    """
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        topk_values, topk_indices = torch.topk(
            input=x,
            k=self.k,
            dim=-1
        )

        active_neurons = topk_values.relu()
        out = torch.zeros_like(x)
        out.scatter_(
            dim=-1,
            index=topk_indices,
            src=active_neurons
        )
        return out
    
class TopKSAEConfig(SAEConfig):
    k: int = 20
    rescale_acts_by_decoder_norm: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"
    

class TopKSAETrainingConfig(TrainingSAEConfig):
    k: int = 20
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"
    

class TopKSAE(SAE):

    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSAEConfig):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the forward pass.
        """
        super().__init__(cfg)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopKActivation(self.cfg.k)

    
    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        sae_in = self.process_sae_in(x)
        hidden_pre_activations = einops.einsum(
            sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_enc
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre_activations = hidden_pre_activations * self.W_dec.norm(dim=-1)
        return self.activation_fn(hidden_pre_activations)

    
    def decode(
        self,
        features: Float[torch.Tensor, "... d_sae"],
    ) -> Float[torch.Tensor, "... d_in"]:
        if self.cfg.rescale_acts_by_decoder_norm:
            features = features / self.W_dec.norm(dim=-1)
        
        sae_out_pre = einops.einsum(
            features, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec
        sae_out = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out


    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Unsafe."
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)


class TopKTrainingSAE(TrainingSAE):
    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSAETrainingConfig):
        super().__init__(cfg)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        sae_in = self.process_sae_in(x)
        hidden_pre_activations = einops.einsum(
            sae_in, self.W_enc, "... d_in, d_in d_sae -> ... d_sae"
        ) + self.b_enc

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre_activations = hidden_pre_activations * self.W_dec.norm(dim=-1)

        features = self.activation_fn(hidden_pre_activations)
        return features, hidden_pre_activations
    
    @override
    def decode(
        self,
        features: Float[torch.Tensor, "... d_sae"],
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space.
        """
        if self.cfg.rescale_acts_by_decoder_norm:
            features = features * (1 / self.W_dec.norm(dim=-1))

        sae_out = einops.einsum(
            features, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        return sae_out
    
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        features = self.encode(x)
        sae_out = self.decode(features)
        return sae_out
    

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        topk_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": topk_loss}
    
    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate TopK auxiliary loss.

        This auxiliary loss encourages dead neurons to learn useful features by having
        them reconstruct the residual error from the live neurons. This aims at preventing neuron death.

        0) Check if neurons are dead -> if no, just return
        1) Get residual
        2) Take number of dead neurons and predict the residual
        3) Calculate aux loss (scale by coeff and scale)
        """
        if dead_neuron_mask is None or (int(dead_neuron_mask.sum()) == 0):
            return sae_out.new_tensor(0.0)
        
        num_dead = int(dead_neuron_mask.sum())
        residual = (sae_in - sae_out).detach()
        aux_k = sae_in.shape[-1] // 2 # heuristic from Eleuther AI

        scale = min(num_dead / aux_k, 1.0)
        aux_k = min(aux_k, num_dead)

        # Don't include living latents in this loss
        auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
        # Top-k dead latents
        auxk_topk = auxk_latents.topk(aux_k, sorted=False)
        # Set the activations to zero for all but the top k_aux dead latents
        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
        
        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss
    

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        if self.cfg.rescale_acts_by_decoder_norm:
            _fold_norm_topk(
                W_enc=state_dict["W_enc"],
                b_enc=state_dict["b_enc"],
                W_dec=state_dict["W_dec"],
            )

    
    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Unnsafe."
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopKActivation(self.cfg.k)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}



def _fold_norm_topk(
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
) -> None:
    W_dec_norm = W_dec.norm(dim=-1)
    b_enc.data = b_enc.data * W_dec_norm
    W_dec_norms = W_dec_norm.unsqueeze(1)
    W_dec.data = W_dec.data / W_dec_norms
    W_enc.data = W_enc.data * W_dec_norms.T



def _init_weights_topk(
    sae: SAE[TopKSAEConfig] | TrainingSAE[TopKSAETrainingConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )