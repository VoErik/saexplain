from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from typing_extensions import override

from src.sae.jumprelu import JumpReLUSAEConfig
from src.sae.base import SAEConfig, TrainStepInput, TrainStepOutput
from src.sae.topk import TopKTrainingSAE, TopKSAETrainingConfig


class BatchTopK(nn.Module):
    """BatchTopK activation function"""

    def __init__(
        self,
        k: float,
    ):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        flat_acts = acts.flatten()
        # Calculate total number of samples across all non-feature dimensions
        num_samples = acts.shape[:-1].numel()
        acts_topk_flat = torch.topk(flat_acts, int(self.k * num_samples), dim=-1)
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, acts_topk_flat.indices, acts_topk_flat.values)
            .reshape(acts.shape)
        )
    
@dataclass
class BatchTopKSAETrainingConfig(TopKSAETrainingConfig):
    k: float = 100  # type: ignore[assignment]
    topk_threshold_lr: float = 0.01

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return JumpReLUSAEConfig
    

class BatchTopKTrainingSAE(TopKTrainingSAE):
    """
    Global Batch TopK Training SAE

    This SAE will maintain the k on average across the batch, rather than enforcing the k per-sample as in standard TopK.

    BatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    topk_threshold: torch.Tensor
    cfg: BatchTopKSAETrainingConfig  # type: ignore[assignment]

    def __init__(self, cfg: BatchTopKSAETrainingConfig):
        super().__init__(cfg)

        self.register_buffer(
            "topk_threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return BatchTopK(self.cfg.k)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        self.update_topk_threshold(output.feature_acts)
        output.metrics["topk_threshold"] = self.topk_threshold
        return output

    @torch.no_grad()
    def update_topk_threshold(self, acts_topk: torch.Tensor) -> None:
        positive_mask = acts_topk > 0
        lr = self.cfg.topk_threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(self.topk_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = (
                    acts_topk[positive_mask].min().to(self.topk_threshold.dtype)
                )
                self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_positive

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        # turn the topk threshold into jumprelu threshold
        topk_threshold = state_dict.pop("topk_threshold").item()
        state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold