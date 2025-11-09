import torch
import torch.nn.functional as F
import wandb

from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Any, Dict

from sae.core import SAE
from sae.utils.scaler import ActivationScaler
from sae.utils.misc import filter_valid_dataclass_fields


@torch.no_grad()
def calculate_activation_coherence(feature_activations: torch.Tensor) -> Dict[str, float]:
    """
    Checks redundancy of features based on their activation correlation.
    Assumes feature_activations is a CPU tensor of shape [n_samples, d_sae]
    """
    centered_activations = feature_activations - feature_activations.mean(dim=0, keepdim=True)
    cov_matrix = (centered_activations.T @ centered_activations) / (len(feature_activations) - 1)
    std = torch.sqrt(torch.diag(cov_matrix))
    correlation = cov_matrix / (std.unsqueeze(1) @ std.unsqueeze(0) + 1e-12)
    correlation.fill_diagonal_(0)
    
    return {
        "metrics/eval_worst_activation_coherence": correlation.abs().max().item(),
        "metrics/eval_avg_activation_coherence": correlation.abs().mean().item()
    }

@torch.no_grad()
def calculate_decoder_coherence(W_dec: torch.Tensor) -> Dict[str, float]:
    """
    Checks redundancy of dictionary atoms (decoder vectors).
    """
    W_dec_cpu = W_dec.detach().cpu()
    W = W_dec_cpu / (W_dec_cpu.norm(dim=1, keepdim=True) + 1e-12)
    G = W @ W.t()
    G.fill_diagonal_(0)
    return {
        "metrics/eval_worst_decoder_coherence": G.abs().max().item(),
        "metrics/eval_avg_decoder_coherence": G.abs().mean().item(),
    }

@torch.no_grad()
def calculate_gini_coefficient(feature_activations: torch.Tensor, eps: float = 1e-10) -> Dict[str, Any]:
    """
    Calculates the Gini coefficient as a proxy for monosemanticity (selectivity).
    """
    n_samples = feature_activations.shape[0]
    if n_samples == 0:
        return {"metrics/eval_mean_gini": 0.0, "plots/eval_gini_histogram": wandb.Histogram([])}

    all_gini = []
    # Only calculate for living features
    living_feature_indices = torch.where(feature_activations.abs().sum(dim=0) > eps)[0]

    if len(living_feature_indices) == 0:
        return {"metrics/eval_mean_gini": 0.0, "plots/eval_gini_histogram": wandb.Histogram([])}

    for j in living_feature_indices:
        acts_j = feature_activations[:, j].abs()
        
        # Sort values
        sorted_acts, _ = torch.sort(acts_j)
        # Calculate cumulative sum
        cum_acts = torch.cumsum(sorted_acts, dim=0)
        # Calculate Lorenz curve area (B)
        lorenz_area = cum_acts.sum() / (n_samples * cum_acts.sum() + eps)
        # Gini = 1 - 2*B
        gini = 1.0 - 2.0 * lorenz_area
        all_gini.append(gini.item())
    
    if not all_gini:
        return {"metrics/eval_mean_gini": 0.0, "plots/eval_gini_histogram": wandb.Histogram([])}
    
    all_gini_tensor = torch.tensor(all_gini, device=feature_activations.device) # device is 'cpu'
    
    return {
        "metrics/eval_mean_gini": all_gini_tensor.mean().item(),
        "plots/eval_gini_histogram": wandb.Histogram(all_gini_tensor.numpy())
    }

@torch.no_grad()
def calculate_class_entropy(feature_activations: torch.Tensor, labels: torch.Tensor, eps: float = 1e-10) -> Dict[str, Any]:
    """
    Calculates the entropy of class distributions for each feature.
    Assumes feature_activations [n_samples, d_sae] and labels [n_samples] are CPU tensors.
    """
    n_samples, d_sae = feature_activations.shape
    
    if n_samples != len(labels):
        print(f"Warning: Mismatch in activation samples ({n_samples}) and labels ({len(labels)}). Skipping class entropy.")
        return {"metrics/eval_mean_class_entropy": 0.0, "plots/eval_class_entropy_histogram": wandb.Histogram([])}
    
    num_classes = labels.max().item() + 1
    all_entropy = []
    
    living_feature_indices = torch.where(feature_activations.abs().sum(dim=0) > eps)[0]
    
    if len(living_feature_indices) == 0:
        return {"metrics/eval_mean_class_entropy": 0.0, "plots/eval_class_entropy_histogram": wandb.Histogram([])}

    for j in living_feature_indices:
        active_mask = feature_activations[:, j] > 0
        if active_mask.sum() == 0:
            continue # Feature is alive but didn't fire > 0
        
        active_labels = labels[active_mask]
        if len(active_labels) == 0:
            continue
        
        # Get class distribution for this feature
        class_counts = torch.bincount(active_labels, minlength=num_classes).float()
        class_probs = class_counts / (class_counts.sum() + eps)
        
        # Calculate entropy H = -sum(p * log(p))
        entropy = -torch.sum(class_probs * torch.log(class_probs + eps))
        all_entropy.append(entropy.item())
        
    if not all_entropy:
        return {"metrics/eval_mean_class_entropy": 0.0, "plots/eval_class_entropy_histogram": wandb.Histogram([])}
    
    all_entropy_tensor = torch.tensor(all_entropy, device=feature_activations.device)
    
    return {
        "metrics/eval_mean_class_entropy": all_entropy_tensor.mean().item(),
        "plots/eval_class_entropy_histogram": wandb.Histogram(all_entropy_tensor.numpy())
    }



@dataclass
class EvaluationConfig:
    eval_batch_size: int = 1024
    num_workers: int = 4
    device: str = "cuda"
    run_expensive_metrics: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res


class SAEEvaluator:
    
    def __init__(self, cfg: EvaluationConfig):
        self.cfg = cfg

    @torch.no_grad()
    def __call__(
        self, 
        sae: SAE, 
        val_dataset: Dataset, 
        activation_scaler: ActivationScaler
    ) -> Dict[str, Any]:
        """
        Runs a full evaluation pass on the validation dataset.
        This method is called by the SAETrainer.
        """
        
        print(f"\nRunning evaluation on {len(val_dataset)} samples...")
        
        sae.eval()
        d_in = sae.cfg.d_in
        d_sae = sae.cfg.d_sae
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )
        
        total_samples = 0
        total_mse_loss_sum = 0.0  # Sum of MSE * (B * D_in)
        total_l2_loss_sum = 0.0   # Sum of L2 loss per token
        total_variance_sum = 0.0  # Sum of variance per token
        total_l0_sum = 0.0        # Sum of L0 norms
        
        running_feature_frequencies = torch.zeros(d_sae, device=self.cfg.device)

        all_feature_acts_cpu = []
        all_labels_cpu = []
        
        has_labels = False

        for batch_raw in tqdm(val_loader, desc="Evaluation"):
            
            if len(batch_raw) == 2:
                batch, labels = batch_raw
                labels = labels.to(self.cfg.device)
                if self.cfg.run_expensive_metrics:
                    all_labels_cpu.append(labels.detach().cpu())
                has_labels = True
            else:
                batch = batch_raw
                labels = None
            
            batch = batch.to(self.cfg.device)
            scaled_batch = activation_scaler(batch)
            n_samples_in_batch = scaled_batch.shape[0]
            total_samples += n_samples_in_batch
            
            # Run SAE forward pass
            feature_acts, _ = sae.encode_with_hidden_pre(scaled_batch)
            sae_out = sae.decode(feature_acts)
                        
            # 1. MSE Loss
            total_mse_loss_sum += F.mse_loss(sae_out, scaled_batch, reduction='sum').item()
            
            # 2. Explained Variance
            per_token_l2_loss = (sae_out - scaled_batch).pow(2).sum(dim=-1) # Shape [B]
            total_variance = (scaled_batch - scaled_batch.mean(0)).pow(2).sum(dim=-1) # Shape [B]
            total_l2_loss_sum += per_token_l2_loss.sum().item()
            total_variance_sum += total_variance.sum().item()

            # 3. L0 Norm
            total_l0_sum += (feature_acts > 0).float().sum().item()
            
            # 4. Feature Frequencies
            running_feature_frequencies += (feature_acts > 0).float().sum(dim=0)
            
            if self.cfg.run_expensive_metrics:
                all_feature_acts_cpu.append(feature_acts.detach().cpu())

        metrics = {}
        
        # 1. Core Metrics
        metrics["losses/eval_loss"] = total_mse_loss_sum / (total_samples * d_in)
        metrics["metrics/eval_l0"] = total_l0_sum / total_samples
        
        mean_l2_loss = total_l2_loss_sum / total_samples
        mean_variance = total_variance_sum / total_samples
        if mean_variance > 1e-9:
            metrics["metrics/eval_explained_variance"] = (1.0 - mean_l2_loss / mean_variance)
        else:
            metrics["metrics/eval_explained_variance"] = 0.0 # Avoid division by zero
        
        # 2. Aggregate Sparsity
        feature_frequencies = (running_feature_frequencies / total_samples).cpu() # Move to CPU
        log_frequencies = torch.log10(feature_frequencies + 1e-10).numpy()
        
        metrics["sparsity/eval_dead_features"] = (feature_frequencies == 0).sum().item()
        metrics["metrics/eval_mean_log10_feature_sparsity"] = log_frequencies.mean().item()
        metrics["plots/eval_feature_density"] = wandb.Histogram(log_frequencies)

        # 3. Decoder Coherence
        metrics.update(calculate_decoder_coherence(sae.W_dec))

        # 4. Expensive
        if self.cfg.run_expensive_metrics:
            print("Running expensive metrics (Gini, Coherence, Entropy)...")
            all_feature_acts = torch.cat(all_feature_acts_cpu, dim=0)
            
            print("Calculating Activation Coherence")
            metrics.update(calculate_activation_coherence(all_feature_acts))
            print("Calculating Gini Coefficient")
            metrics.update(calculate_gini_coefficient(all_feature_acts))

            if has_labels:
                all_labels = torch.cat(all_labels_cpu, dim=0)
                print("Calculating Class Entropy")
                metrics.update(calculate_class_entropy(all_feature_acts, all_labels))
        
        print(f"Evaluation complete. Eval Loss: {metrics['losses/eval_loss']:.4f}")
        
        return metrics