from jaxtyping import Float
import math
import torch
from tqdm import tqdm
from typing import Dict
from src.models.sae.activation_store import VisionActivationStore

###################################################################################
################################## METRICS ########################################
###################################################################################

def calculate_l0_sparsity(
        feature_acts: Float[torch.Tensor, "*batch d_sae"],
        abs_threshold: float = 1e-7
) -> Float[torch.Tensor, ""]:
    """
    Calculates the L0 sparsity.
    """
    if feature_acts.ndim < 1:
        return torch.tensor(0.0, device=feature_acts.device)
    if feature_acts.ndim == 1:  # Single feature vector
        num_active_features = (torch.abs(feature_acts) > abs_threshold).float().sum()
        return num_active_features
    else:  # Batch of feature vectors
        num_active_features_per_item = (torch.abs(feature_acts) > abs_threshold).float().sum(dim=-1)
        return num_active_features_per_item.mean()


def calculate_mean_log_feature_density(
        feature_activity_counts: Float[torch.Tensor, "d_sae"],
        num_total_items: int,
        epsilon: float = 1e-10
) -> Float[torch.Tensor, ""]:
    """
    Calculates the mean of log10 feature densities.
    Density = activity_count / num_total_items.
    """
    if num_total_items <= 0:
        return torch.tensor(float('nan'), device=feature_activity_counts.device)
    if feature_activity_counts.ndim != 1:
        raise ValueError("feature_activity_counts must be 1D (d_sae).")

    feature_densities = feature_activity_counts / num_total_items
    log_densities = torch.log10(feature_densities + epsilon)
    return log_densities.mean()


def calculate_decoder_coherence(W_dec: torch.Tensor) -> dict[str, float]:
    """
    Checks redundancy of dictionary atoms.
    """
    W = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-12)
    G = W @ W.t()
    G.fill_diagonal_(0)
    return {
        "worst_decoder_coherence": G.abs().max().item(),
        "avg_decoder_coherence": G.abs().mean().item(),
    }

def calculate_hoyer_sparsity(
        feature_acts: torch.Tensor,
        epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Calculates Hoyer's sparsity for a batch of feature activations.

    Args:
        feature_acts: Tensor of shape (*batch_dims, d_sae)

    Returns:
        A single tensor containing the SUM of Hoyer's sparsity values for the batch.
    """
    d_sae = feature_acts.shape[-1]
    flat_acts = feature_acts.reshape(-1, d_sae) # (B, P, D) -> (B*P, D)

    l1_norms = flat_acts.abs().sum(dim=-1)
    l2_norms = flat_acts.pow(2).sum(dim=-1).sqrt()

    sqrt_d = d_sae**0.5
    hoyer_values = (sqrt_d - l1_norms / (l2_norms + epsilon)) / (sqrt_d - 1 + epsilon)

    return hoyer_values.sum()

###################################################################################
################################## EVALUATOR ######################################
###################################################################################

class SAEEvaluator:
    def __init__(
            self,
            sae_model: 'SAE',
            eval_activation_store: VisionActivationStore,
            device: torch.device,
            eval_n_batches: int | None = None,
    ):
        self.sae_model = sae_model
        self.eval_activation_store = eval_activation_store
        self.device = device
        self.eval_n_batches = eval_n_batches

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.sae_model.to(self.device)
        self.sae_model.eval()

        num_batches_to_process = len(self.eval_activation_store)
        if self.eval_n_batches is not None and self.eval_n_batches > 0:
            num_batches_to_process = min(num_batches_to_process, self.eval_n_batches)

        if num_batches_to_process == 0:
            print("Warning: No batches to process for evaluation.")
            return {
                k: float('nan') for k in ["eval_mse", "eval_l0_sparsity",
                                          "eval_explained_variance", "hoyer_sparsity",
                                          "eval_mean_log_feature_density", "eval_dead_features_on_eval_set",
                                          "worst_decoder_coherence", "avg_decoder_coherence"]
            }

        total_items_processed = 0
        total_l0_sparsity = 0.0
        total_hoyer_sparsity = 0.0
        total_sum_squared_error = 0.0
        total_variance_explained_num = 0.0
        total_variance_explained_den = 0.0
        k = self.sae_model.cfg.activation_fn_kwargs.k \
            if self.sae_model.cfg.activation_fn_str == "topk" \
            else math.floor(
            self.sae_model.cfg.d_sae * 0.05 # use 5% if no k is given via config
        )
        topk_counts = torch.zeros(self.sae_model.cfg.d_sae).to(self.device)


        feature_activity_counts = torch.zeros(self.sae_model.cfg.d_sae, device=self.device)

        batch_iterator = iter(self.eval_activation_store)
        for i in tqdm(range(num_batches_to_process), desc="Evaluating SAE"):
            try:
                batch_sae_in = next(batch_iterator).to(self.device)
            except StopIteration:
                break

            # Determine number of items (patches) in this batch
            num_items_in_batch = batch_sae_in.nelement() / batch_sae_in.shape[-1]
            total_items_processed += num_items_in_batch

            feature_acts, _ = self.sae_model.encode_with_hidden_pre(batch_sae_in)
            sae_out = self.sae_model.decode(feature_acts)

            # MSE
            total_sum_squared_error += torch.sum((batch_sae_in - sae_out) ** 2)

            # L0 Sparsity
            total_l0_sparsity += (feature_acts > 0).sum()

            # Hoyer Sparsity
            total_hoyer_sparsity += calculate_hoyer_sparsity(feature_acts)

            # Explained Variance
            total_variance_explained_num += torch.sum((batch_sae_in - sae_out) ** 2)
            total_variance_explained_den += torch.sum((batch_sae_in - batch_sae_in.mean()) ** 2)

            # Feature Activity
            feature_activity_counts += (feature_acts > 0).sum(dim=tuple(range(feature_acts.ndim - 1)))

            # TopK Entropy
            topk_vals, topk_idxs = torch.topk(feature_acts, k=k, dim=-1)
            flat_topk_idxs = topk_idxs.reshape(-1, k)
            topk_counts += torch.bincount(
                flat_topk_idxs.view(-1), minlength=self.sae_model.cfg.d_sae).to(self.device)

        metrics_dict: Dict[str, float] = {}

        # Finalize averages by dividing by total items
        final_mse = total_sum_squared_error / (total_items_processed * self.sae_model.cfg.d_in)
        final_l0 = total_l0_sparsity / total_items_processed
        final_hoyer = total_hoyer_sparsity / total_items_processed

        if total_variance_explained_den > 0:
            final_ev = (1 - total_variance_explained_num / total_variance_explained_den).item()
        else:
            final_ev = float('nan')

        if topk_counts.sum() > 0:
            probs = topk_counts / topk_counts.sum()
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        else:
            entropy = float('nan')

        metrics_dict["eval_mse"] = final_mse.item()
        metrics_dict["eval_l0_sparsity"] = final_l0.item()
        metrics_dict["eval_explained_variance"] = final_ev
        metrics_dict["hoyer_sparsity"] = final_hoyer.item()
        metrics_dict["eval_topk_entropy"] = entropy

        # Feature density metrics
        metrics_dict["eval_mean_log_feature_density"] = calculate_mean_log_feature_density(
            feature_activity_counts.cpu(), total_items_processed
        ).item()

        # Count dead features
        metrics_dict["eval_dead_features_on_eval_set"] = (feature_activity_counts == 0).sum().item()

        # Decoder coherence (proxy for purity of SAE)
        coherence_dict = calculate_decoder_coherence(self.sae_model.W_dec.to(self.device))
        metrics_dict["worst_decoder_coherence"] = coherence_dict["worst_decoder_coherence"]
        metrics_dict["avg_decoder_coherence"] = coherence_dict["avg_decoder_coherence"]

        self.sae_model.train()
        return metrics_dict