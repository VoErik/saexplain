import warnings
from dataclasses import dataclass, field

import einops
import torch
from jaxtyping import Float
from typing_extensions import override

from src.sae.batchtopk import (
    BatchTopKTrainingSAE,
    BatchTopKSAETrainingConfig,
)
from src.sae.base import TrainStepInput, TrainStepOutput


@dataclass
class MatryoshkaBatchTopKSAETrainingConfig(BatchTopKSAETrainingConfig):
    """
    Configuration class for training a MatryoshkaBatchTopKTrainingSAE.

    [Matryoshka SAEs](https://arxiv.org/pdf/2503.17547) use a series of nested reconstruction
    losses of different widths during training to avoid feature absorption. This also has a
    nice side-effect of encouraging higher-frequency features to be learned in earlier levels.
    However, this SAE has more hyperparameters to tune than standard BatchTopK SAEs, and takes
    longer to train due to requiring multiple forward passes per training step.

    After training, MatryoshkaBatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        matryoshka_widths (list[int]): The widths of the matryoshka levels. Defaults to an empty list.
    """

    matryoshka_widths: list[int] = field(default_factory=list)

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka"


class MatryoshkaBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    Global Batch TopK Training SAE

    This SAE will maintain the k on average across the batch, rather than enforcing the k per-sample as in standard TopK.

    BatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    cfg: MatryoshkaBatchTopKSAETrainingConfig  # type: ignore[assignment]

    def __init__(
        self, cfg: MatryoshkaBatchTopKSAETrainingConfig
    ):
        super().__init__(cfg)
        _validate_matryoshka_config(cfg)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        base_output = super().training_forward_pass(step_input)
        inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)
        # the outer matryoshka level is the base SAE, so we don't need to add an extra loss for it
        for width in self.cfg.matryoshka_widths[:-1]:
            inner_reconstruction = self._decode_matryoshka_level(
                base_output.feature_acts, width, inv_W_dec_norm
            )
            inner_mse_loss = (
                self.mse_loss_fn(inner_reconstruction, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            base_output.losses[f"inner_mse_loss_{width}"] = inner_mse_loss
            base_output.loss = base_output.loss + inner_mse_loss
        return base_output

    def _decode_matryoshka_level(
        self,
        features: Float[torch.Tensor, "... d_sae"],
        width: int,
        inv_W_dec_norm: torch.Tensor,
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space for a matryoshka level
        """
        inner_features = features[:, :width]
        if self.cfg.rescale_acts_by_decoder_norm:
            inner_features = inner_features * inv_W_dec_norm[:width]
        
        sae_out = einops.einsum(
            inner_features, self.W_dec[:width], "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec
        
        sae_out = self.run_time_activation_norm_fn_out(sae_out)
        return sae_out


def _validate_matryoshka_config(cfg: MatryoshkaBatchTopKSAETrainingConfig) -> None:
    if cfg.matryoshka_widths[-1] != cfg.d_sae:
        warnings.warn(
            "WARNING: The final matryoshka level width is not set to cfg.d_sae. "
            "A final matryoshka level of width=cfg.d_sae will be added."
        )
        cfg.matryoshka_widths.append(cfg.d_sae)

    for prev_width, curr_width in zip(
        cfg.matryoshka_widths[:-1], cfg.matryoshka_widths[1:]
    ):
        if prev_width >= curr_width:
            raise ValueError("cfg.matryoshka_widths must be strictly increasing.")
    if len(cfg.matryoshka_widths) == 1:
        warnings.warn(
            "WARNING: You have only set one matryoshka level. This is equivalent to using a standard BatchTopK SAE."
        )