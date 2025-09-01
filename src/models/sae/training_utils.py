import torch.nn.functional as F
from jaxtyping import Float

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import Optimizer

def mse_loss(
        reconstruction: Float[torch.Tensor, "*batch d_in"],
        original: Float[torch.Tensor, "*batch d_in"]
) -> Float[torch.Tensor, ""]:
    """
    Calculates the Mean Squared Error (MSE) between the reconstructed output and the original input.
    The loss is averaged over all dimensions (batch and d_in).
    """
    if reconstruction.shape != original.shape:
        raise ValueError(f"Shape mismatch: reconstruction {reconstruction.shape}, original {original.shape}")
    return F.mse_loss(reconstruction, original, reduction='mean')


def l1_sparsity_loss(
        feature_activations: Float[torch.Tensor, "*batch d_sae"],
        l1_coefficient: float
) -> Float[torch.Tensor, ""]:
    """
    Calculates the L1 sparsity loss on the feature activations.
    The L1 norm is summed over the feature dimension (d_sae) and then averaged over the batch.
    Args:
        feature_activations: The activations of the SAE's hidden layer.
        l1_coefficient: The weight for this L1 penalty.
    Returns:
        The L1 sparsity loss value.
    """
    if l1_coefficient < 0:
        raise ValueError("L1 coefficient must be non-negative.")
    if l1_coefficient == 0:
        return torch.tensor(0.0, device=feature_activations.device, dtype=feature_activations.dtype)

    sum_abs_acts = torch.abs(feature_activations).sum(dim=-1)
    mean_sum_abs_acts = sum_abs_acts.mean()

    return l1_coefficient * mean_sum_abs_acts

def lp_sparsity_loss(
        feature_activations: Float[torch.Tensor, "*batch d_sae"],
        lp_norm: float,
        l1_coefficient: float
) -> Float[torch.Tensor, ""]:
    """
    Calculates the LP sparsity loss on the feature activations.

    Args:
        feature_activations: The activations of the SAE's hidden layer.
        lp_norm: The norm of the LP penalty.
        l1_coefficient: The weight for this L1 penalty.

    Returns:
        LP sparsity loss.
    """
    if lp_norm <= 0:
        raise ValueError("LP norm must be positive.")
    if l1_coefficient == 0:
        return torch.tensor(0.0, device=feature_activations.device)

    # Calculate Lp norm per hidden vector, then average
    lp_norms_per_vector = torch.norm(feature_activations, p=lp_norm, dim=-1)
    mean_lp_norm = lp_norms_per_vector.mean()
    return l1_coefficient * mean_lp_norm


def topk_auxiliary_loss(
        sae_in: Float[torch.Tensor, "*batch d_in"],
        sae_out: Float[torch.Tensor, "*batch d_in"],
        hidden_pre_acts: Float[torch.Tensor, "*batch d_sae"],
        decode_fn: callable,  # Function: (Float[torch.Tensor, "*batch d_sae"]) -> Float[torch.Tensor, "*batch d_in"]
        dead_neuron_mask: Float[torch.Tensor, "d_sae"],
        k_aux: int,
) -> Float[torch.Tensor, ""]:
    """
    Calculates the auxiliary loss for TopK SAEs to encourage dead neurons
    to predict the reconstruction error.

    Args:
        sae_in: Original input to the SAE.
        sae_out: Reconstruction from the SAE's main forward pass.
        hidden_pre_acts: Pre-activation values of the SAE's hidden layer.
        decode_fn: A callable (e.g., partial SAE.decode or lambda) that decodes feature activations.
        dead_neuron_mask: Boolean tensor indicating dead features.
        k_aux: Number of top dead features to consider for this loss.

    Returns:
        The auxiliary loss value.
    """
    num_dead_neurons = dead_neuron_mask.sum().item()

    if num_dead_neurons == 0 or k_aux == 0:
        return torch.tensor(0.0, device=sae_in.device, dtype=sae_in.dtype)

    actual_k_aux = min(k_aux, int(num_dead_neurons))
    if actual_k_aux == 0:
        return torch.tensor(0.0, device=sae_in.device, dtype=sae_in.dtype)

    residual = (sae_in - sae_out).detach()

    batch_dead_neuron_mask = dead_neuron_mask.unsqueeze(0).expand_as(hidden_pre_acts)

    auxk_latents = torch.where(batch_dead_neuron_mask, hidden_pre_acts,
                               torch.tensor(-float('inf'), device=hidden_pre_acts.device))

    try:
        auxk_top_values, auxk_top_indices = torch.topk(auxk_latents, k=actual_k_aux, dim=-1)
    except RuntimeError as e:
        print(f"RuntimeError in torch.topk for aux loss: {e}. num_dead={num_dead_neurons}, actual_k_aux={actual_k_aux}")
        print(f"auxk_latents non-inf check: {torch.isfinite(auxk_latents[batch_dead_neuron_mask]).sum()}")
        return torch.tensor(0.0, device=sae_in.device, dtype=sae_in.dtype)

    # Create feature activations using only these top k_aux dead neurons' pre-activations
    auxk_feature_acts = torch.zeros_like(hidden_pre_acts)
    auxk_feature_acts.scatter_(-1, auxk_top_indices, auxk_top_values)
    auxk_feature_post_acts = F.relu(auxk_feature_acts)
    recons_of_residual_by_dead = decode_fn(auxk_feature_post_acts)

    aux_loss = mse_loss(recons_of_residual_by_dead, residual)

    # Scale the loss as suggested in some implementations (e.g. EleutherAI SAE paper appendix)
    # "The coefficient for this auxiliary loss is scaled by min(1, N_dead / k_aux )"
    # Here, k_aux is the target number of features for the aux loss, not total dictionary size.
    # The sae-lens code has: `scale = min(num_dead / k_aux, 1.0)`
    # where their k_aux is a target number (e.g., d_in // 2), not number of dead neurons.
    scale_factor = min(float(num_dead_neurons) / k_aux, 1.0)

    return aux_loss * scale_factor



class L1Scheduler:
    """
    Scheduler for the L1 penalty coefficient.
    It can linearly warm up the L1 coefficient from an initial value to a final value.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            l1_warm_up_steps: int,
            total_steps: int,
            final_l1_coefficient: float,
            initial_l1_coefficient: float = 0.0,
    ):
        if not isinstance(l1_warm_up_steps, int) or l1_warm_up_steps < 0:
            raise ValueError("l1_warm_up_steps must be a non-negative integer.")
        if not isinstance(total_steps, int) or total_steps <= 0:
            if l1_warm_up_steps > total_steps:
                raise ValueError("l1_warm_up_steps cannot be greater than total_steps.")
        if not isinstance(final_l1_coefficient, float) or final_l1_coefficient < 0:
            raise ValueError("final_l1_coefficient must be a non-negative float.")
        if not isinstance(initial_l1_coefficient, float) or initial_l1_coefficient < 0:
            raise ValueError("initial_l1_coefficient must be a non-negative float.")

        self.l1_warm_up_steps = l1_warm_up_steps
        self.total_steps = total_steps
        self.final_l1_coefficient = final_l1_coefficient
        self.initial_l1_coefficient = initial_l1_coefficient
        self.current_step = 0
        self._current_l1_coefficient = initial_l1_coefficient

    def step(self):
        """Advance the scheduler by one step."""
        self.current_step += 1
        if self.l1_warm_up_steps == 0:  # No warmup, jump to final
            self._current_l1_coefficient = self.final_l1_coefficient
        elif self.current_step <= self.l1_warm_up_steps:
            self._current_l1_coefficient = self.initial_l1_coefficient + \
                                           (self.final_l1_coefficient - self.initial_l1_coefficient) * \
                                           (self.current_step / self.l1_warm_up_steps)
        else:
            self._current_l1_coefficient = self.final_l1_coefficient

        if self.initial_l1_coefficient < self.final_l1_coefficient:
            self._current_l1_coefficient = min(self._current_l1_coefficient, self.final_l1_coefficient)
        else:
            self._current_l1_coefficient = max(self._current_l1_coefficient, self.final_l1_coefficient)

    @property
    def current_l1_coefficient(self) -> float:
        """Returns the current L1 coefficient."""
        return self._current_l1_coefficient

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "current_l1_coefficient": self._current_l1_coefficient,
            "l1_warm_up_steps": self.l1_warm_up_steps,
            "total_steps": self.total_steps,
            "final_l1_coefficient": self.final_l1_coefficient,
            "initial_l1_coefficient": self.initial_l1_coefficient,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self._current_l1_coefficient = state_dict["current_l1_coefficient"]


def get_lr_scheduler(
        optimizer: Optimizer,
        scheduler_name: str,
        warm_up_steps: int,
        decay_steps: int | None = None,
        training_steps: int | None = None,
        lr_end: float = 0.0,
        num_cycles: int | None = None,
):
    """
    Fetches a learning rate scheduler.

    Args:
        optimizer: The optimizer.
        scheduler_name: Name of the scheduler ("linear_warmup_decay", "cosine_warmup_restarts").
        warm_up_steps: Number of warm-up steps.
        decay_steps: Number of decay steps after warmup (for linear).
        training_steps: Total training steps (alternative to decay_steps for some schedulers).
        lr_end: The final learning rate after decay (for linear).
        num_cycles: Number of cycles for cosine annealing with restarts.
    """
    if training_steps is None and decay_steps is not None:
        training_steps = warm_up_steps + decay_steps
    elif training_steps is None and decay_steps is None and scheduler_name != "reduce_on_plateau":
        raise ValueError("Either training_steps or decay_steps (for linear) must be provided.")

    if scheduler_name.lower() == "linear_warmup_decay":
        if training_steps is None:
            raise ValueError("training_steps must be provided for linear_warmup_decay")

        def lr_lambda(current_step: int):
            if current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            progress = float(current_step - warm_up_steps) / float(max(1, training_steps - warm_up_steps))
            if progress > 1.0: progress = 1.0
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_name.lower() == "cosine_warmup_restarts":
        if training_steps is None:
            raise ValueError("training_steps must be provided for cosine_warmup_restarts")
        if num_cycles is None:
            num_cycles = 0.5

        def lr_lambda_cosine(current_step: int):
            if current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            else:
                progress = float(current_step - warm_up_steps) / float(max(1, training_steps - warm_up_steps))
                if progress >= 1.0:
                    return lr_end / optimizer.defaults['lr'] if lr_end > 0 else 0.0
                scale = 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress)).item())
                target_lr_multiple = lr_end / optimizer.defaults['lr']
                return target_lr_multiple + (1.0 - target_lr_multiple) * scale

        return LambdaLR(optimizer, lr_lambda_cosine)


    elif scheduler_name.lower() == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    elif scheduler_name.lower() == "none" or scheduler_name is None:
        return LambdaLR(optimizer, lr_lambda=lambda current_step: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")