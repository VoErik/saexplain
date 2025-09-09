from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from jaxtyping import Float
from torch import nn

from src.models.sae.architectures import (
    TopKSAE,
    StandardSAE,
    GatedSAE,
)
from src.models.sae.config import TrainingSAEConfig
from src.models.sae.core import SAE, TopK
from src.models.sae.training_utils import (
    mse_loss,
    l1_sparsity_loss,
    topk_auxiliary_loss,
    L1Scheduler,
    get_lr_scheduler
)
from src.models.sae.activation_store import VisionActivationStore
from src.models.sae.eval import SAEEvaluator

import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
from pathlib import Path
import math

@dataclass
class TrainStepOutput:
    total_loss: Float[torch.Tensor, ""]
    mse_loss: Float[torch.Tensor, ""]
    l1_sparsity_loss: Optional[Float[torch.Tensor, ""]] = None
    aux_loss: Optional[Float[torch.Tensor, ""]] = None
    sae_in: Optional[Float[torch.Tensor, "*batch d_in"]] = None
    sae_out: Optional[Float[torch.Tensor, "*batch d_in"]] = None
    feature_acts: Optional[Float[torch.Tensor, "*batch d_sae"]] = None
    hidden_pre_acts: Optional[Float[torch.Tensor, "*batch d_sae"]] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total_loss": self.total_loss.item(),
            "mse_loss": self.mse_loss.item(),
        }
        if self.l1_sparsity_loss is not None:
            d["l1_sparsity_loss"] = self.l1_sparsity_loss.item()
        if self.aux_loss is not None:
            d["aux_loss"] = self.aux_loss.item()
        return d

class TrainingSAEBase(SAE, ABC):
    """
    Abstract base class for SAEs specifically designed for training.
    It enforces a training_forward_pass method.
    """
    cfg: TrainingSAEConfig

    def __init__(self, cfg: TrainingSAEConfig):
        super().__init__(cfg)  # BaseSAE.__init__
        if not isinstance(cfg, TrainingSAEConfig):
            raise ValueError(f"{self.__class__.__name__} requires a TrainingSAEConfig instance.")

    @abstractmethod
    def training_forward_pass(
            self,
            sae_in: Float[torch.Tensor, "*batch d_in"],
            dead_neuron_mask: Optional[Float[torch.Tensor, "d_sae"]] = None,  # dead_neuron_mask is boolean
            store_activations: bool = False,
            current_l1_coefficient: float = 0.0,
    ) -> TrainStepOutput:
        pass

class TrainingStandardSAE(TrainingSAEBase, StandardSAE):
    """
    Training version of the StandardSAE.
    """
    cfg: TrainingSAEConfig

    def __init__(self, cfg: TrainingSAEConfig):
        if cfg.architecture not in ["standard", "jumprelu"]:
            raise ValueError("TrainingStandardSAE instantiated with non-'standard' architecture.")
        TrainingSAEBase.__init__(self, cfg)
        StandardSAE.__init__(self, cfg)

    def training_forward_pass(
            self,
            sae_in: Float[torch.Tensor, "*batch d_in"],
            dead_neuron_mask: Optional[Float[torch.Tensor, "d_sae"]] = None,
            store_activations: bool = False,
            current_l1_coefficient: float = 0.0,
    ) -> TrainStepOutput:

        sae_in_device = sae_in.to(device=self.device, dtype=self.dtype)

        feature_acts, hidden_pre_acts = self.encode_with_hidden_pre(sae_in_device)
        sae_out = self.decode(feature_acts)

        current_mse_loss = mse_loss(sae_out, sae_in_device)
        current_l1_loss = l1_sparsity_loss(feature_acts, self.cfg.l1_coefficient)

        total_loss = current_mse_loss + current_l1_loss

        return TrainStepOutput(
            total_loss=total_loss,
            mse_loss=current_mse_loss,
            l1_sparsity_loss=current_l1_loss,
            aux_loss=None,
            sae_in=sae_in_device if store_activations else None,
            sae_out=sae_out if store_activations else None,
            feature_acts=feature_acts if store_activations else None,
            hidden_pre_acts=hidden_pre_acts if store_activations else None
        )

class TrainingTopKSAE(TrainingSAEBase, TopKSAE):
    """
    Training version of the TopKSAE.
    """
    cfg: TrainingSAEConfig

    def __init__(self, cfg: TrainingSAEConfig):
        if cfg.architecture not in ["topk", "batchtopk"]:
            raise ValueError("TrainingTopKSAE instantiated with non-'topk' architecture.")
        TrainingSAEBase.__init__(self, cfg)
        TopKSAE.__init__(self, cfg)

    def training_forward_pass(
            self,
            sae_in: Float[torch.Tensor, "*batch d_in"],
            dead_neuron_mask: Optional[Float[torch.Tensor, "d_sae"]] = None,
            store_activations: bool = False,
            current_l1_coefficient: float = 0.0,
    ) -> TrainStepOutput:
        sae_in_device = sae_in.to(device=self.device, dtype=self.dtype)

        feature_acts, hidden_pre_acts = self.encode_with_hidden_pre(sae_in_device)
        sae_out = self.decode(feature_acts)

        current_mse_loss = mse_loss(sae_out, sae_in_device)
        total_loss = current_mse_loss

        current_aux_loss = None
        if isinstance(self.activation_fn, TopK) and \
                self.cfg.topk_aux_loss_coefficient > 0 and \
                dead_neuron_mask is not None and \
                dead_neuron_mask.sum().item() > 0:

            if self.cfg.aux_k is not None:
                actual_k_aux = self.cfg.aux_k
            else:
                actual_k_aux = self.cfg.d_in // 2

            if actual_k_aux > 0:
                def _decode_features_for_aux(acts_for_aux: torch.Tensor) -> torch.Tensor:
                    return torch.matmul(acts_for_aux, self.W_dec) + self.b_dec

                current_aux_loss = topk_auxiliary_loss(
                    sae_in=sae_in_device,
                    sae_out=sae_out,
                    hidden_pre_acts=hidden_pre_acts,
                    decode_fn=_decode_features_for_aux,
                    dead_neuron_mask=dead_neuron_mask,
                    k_aux=actual_k_aux
                )
                total_loss = total_loss + self.cfg.topk_aux_loss_coefficient * current_aux_loss

        return TrainStepOutput(
            total_loss=total_loss,
            mse_loss=current_mse_loss,
            l1_sparsity_loss=None,
            aux_loss=current_aux_loss,
            sae_in=sae_in_device if store_activations else None,
            sae_out=sae_out if store_activations else None,
            feature_acts=feature_acts if store_activations else None,
            hidden_pre_acts=hidden_pre_acts if store_activations else None
        )

class TrainingGatedSAE(TrainingSAEBase, GatedSAE):
    """ Training version of the DeepMind-style GatedSAE. """
    cfg: TrainingSAEConfig

    def __init__(self, cfg: TrainingSAEConfig):
        if cfg.architecture != "gated":
            raise ValueError("TrainingGatedSAE requires cfg.architecture == 'gated'")
        TrainingSAEBase.__init__(self, cfg)
        GatedSAE.__init__(self, cfg)

    def calculate_aux_loss(
            self,
            sae_in: torch.Tensor,
            current_l1_coefficient: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes:
          - Weighted L1 penalty (gate activations × decoder norms)
          - Auxiliary reconstruction loss (from gate path only)
        """
        if self.cfg.apply_b_dec_to_input:
            sae_in_centered = sae_in - self.b_dec
        else:
            sae_in_centered = sae_in

        # Gate pre-activations
        pi_gate = sae_in_centered @ self.W_enc + self.b_gate
        pi_gate_act = torch.relu(pi_gate)

        # Weighted L1 penalty
        W_dec_norms = self.W_dec.norm(dim=1)  # shape (d_sae,)
        l1_loss = (
                current_l1_coefficient
                * (pi_gate_act * W_dec_norms).sum(dim=-1).mean()
        )

        # Aux reconstruction using only gate path
        via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
        aux_recon_loss = ((via_gate_reconstruction - sae_in) ** 2).sum(dim=-1).mean()

        return l1_loss, aux_recon_loss

    def training_forward_pass(
            self,
            sae_in: Float[torch.Tensor, "*batch d_in"],
            dead_neuron_mask: Optional[Float[torch.Tensor, "d_sae"]] = None,
            store_activations: bool = False,
            current_l1_coefficient: float = 0.0,
    ) -> TrainStepOutput:
        sae_in = sae_in.to(device=self.device, dtype=self.dtype)

        # Encode + decode
        feature_acts, hidden_pre_acts = self.encode_with_hidden_pre(sae_in)
        sae_out = self.decode(feature_acts)

        # Main MSE loss
        mse_loss_val = nn.functional.mse_loss(sae_out, sae_in)

        # Aux losses
        l1_loss_val, aux_recon_loss_val = self.calculate_aux_loss(
            sae_in, current_l1_coefficient
        )

        # Total loss
        total_loss = mse_loss_val + l1_loss_val + aux_recon_loss_val

        return TrainStepOutput(
            total_loss=total_loss,
            mse_loss=mse_loss_val,
            l1_sparsity_loss=l1_loss_val,
            aux_loss=aux_recon_loss_val,
            sae_in=sae_in_device if store_activations else None,
            sae_out=sae_out if store_activations else None,
            feature_acts=feature_acts if store_activations else None,
            hidden_pre_acts=hidden_pre_acts if store_activations else None,
        )

def get_training_sae(architecture: str, cfg: TrainingSAEConfig) -> TrainingSAEBase:
    if architecture in ["topk", "batchtopk"]:
        return TrainingTopKSAE(cfg=cfg)
    elif architecture in ["standard", "jumprelu"]:
        return TrainingStandardSAE(cfg=cfg)
    elif architecture == "gated":
        return TrainingGatedSAE(cfg=cfg)
    else:
        raise NotImplementedError(f"Architecture '{architecture}' not implemented.")

class SAETrainer:
    def __init__(
            self,
            sae_model: TrainingSAEBase,
            activation_store: VisionActivationStore,
            eval_activation_store: VisionActivationStore = None,
            verbose: bool = False,
    ):
        self.sae_model = sae_model
        self.activation_store = activation_store
        self.eval_activation_store = eval_activation_store
        self.config = sae_model.cfg
        self.verbose = verbose

        if not isinstance(self.config, TrainingSAEConfig):
            raise ValueError("SAETrainerBasic requires TrainingSAEConfig.")

        self.device = self.sae_model.device

        # Set up Evaluator
        self.evaluator: SAEEvaluator | None = None
        if self.eval_activation_store is not None:
            self.evaluator = SAEEvaluator(
                sae_model=self.sae_model,
                eval_activation_store=self.eval_activation_store,
                device=self.device,
                eval_n_batches=self.config.eval_n_batches
            )
            print(f"Evaluator initialized. Will evaluate every {self.config.eval_interval_steps} steps "
                  f"using {self.config.eval_n_batches or 'all'} batches from eval store.")
        else:
            print("No evaluation activation store provided. Skipping evaluation during training.")

        # set lr
        self.config.lr = self.config.lr * (self.config.train_batch_size / 256)

        self.optimizer = optim.AdamW(self.sae_model.parameters(), lr=self.config.lr)

        # Initialize LR Scheduler
        if self.config.total_training_steps is None:
            estimated_steps_per_epoch = len(self.activation_store) # num_batches
            self.actual_total_training_steps = estimated_steps_per_epoch
            print(
                f"Warning: config.total_training_steps not set. "
                f"Estimating {self.actual_total_training_steps} for scheduler based on one epoch.")
            if self.config.lr_scheduler_name not in [None, "none", "reduce_on_plateau"]:
                print(
                    f"Consider setting config.total_training_steps for schedulers like {self.config.lr_scheduler_name}")
        else:
            self.actual_total_training_steps = self.config.total_training_steps

        self.lr_scheduler = get_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_name=self.config.lr_scheduler_name if self.config.lr_scheduler_name else "none",
            warm_up_steps=self.config.lr_warm_up_steps,
            training_steps=self.actual_total_training_steps,
            lr_end=self.config.lr * self.config.lr_end_factor,
        )

        # Initialize L1 Scheduler
        self.l1_scheduler = L1Scheduler(
            optimizer=self.optimizer,
            l1_warm_up_steps=self.config.l1_warm_up_steps,
            total_steps=self.actual_total_training_steps,
            final_l1_coefficient=self.config.l1_coefficient,
            initial_l1_coefficient=self.config.initial_l1_coefficient,
        )

        # Dead neuron tracking
        self.n_forward_passes_since_fired = torch.zeros(self.sae_model.cfg.d_sae, device=self.device, dtype=torch.long)
        self.act_freq_scores = torch.zeros(self.sae_model.cfg.d_sae, device=self.device, dtype=torch.float32)
        self.n_seen_tokens_for_sparsity = 0
        self.total_optimizer_steps = 0

        # Checkpointing
        self.checkpoint_counter = 0
        self.checkpoint_save_paths: list[Path] = []

        # W&B
        if self.config.log_to_wandb:
            run_name = self.config.wandb_run_name if self.config.wandb_run_name \
                else f"{self.sae_model.get_name()}_{torch.randint(10000, (1,)).item()}"
            wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.to_dict(),
                name=run_name,
                save_code=False,
            )
            wandb.watch(
                self.sae_model, log_freq=self.config.wandb_log_frequency_steps * 5, log="all"
            )

        self.autocast_dtype = None
        if self.config.autocast_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
            print("Using bfloat16 for mixed precision.")
        elif self.config.autocast_fp16 and torch.cuda.is_available():
            self.autocast_dtype = torch.float16
            print("Using float16 for mixed precision.")

        self.grad_scaler = GradScaler(enabled=(self.autocast_dtype is not None and self.device.type == 'cuda'))
        print(f"GradScaler enabled: {self.grad_scaler.is_enabled()}")

    @property
    def dead_neuron_mask(self) -> torch.Tensor:
        if not hasattr(self.config, 'dead_feature_window'):
            return torch.zeros(self.sae_model.cfg.d_sae, device=self.device, dtype=torch.bool)
        return (self.n_forward_passes_since_fired > self.config.dead_feature_window).bool()

    def _update_dead_neuron_stats(self, feature_acts: torch.Tensor, batch_size: int):
        if feature_acts is None or feature_acts.ndim < 2: return
        with torch.no_grad():
            did_fire_this_step = (feature_acts.abs() > 1e-7).any(dim=(0, 1))
            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire_this_step] = 0
            self.act_freq_scores += (feature_acts.abs() > 1e-7).any(dim=(0, 1)).float()
            self.n_seen_tokens_for_sparsity += batch_size

    def _get_current_l1_coeff_for_loss(self) -> float:
        if self.config.architecture == "topk" and self.config.l1_coefficient == 0:
            return 0.0
        return self.l1_scheduler.current_l1_coefficient

    def _checkpoint_if_needed(self, epoch: int):
        """Saves a model checkpoint if conditions are met."""
        if self.config.n_checkpoints <= 0:
            return

        if self.actual_total_training_steps is None or self.actual_total_training_steps <= 0: return

        checkpoint_interval = math.ceil(self.actual_total_training_steps / self.config.n_checkpoints)
        if checkpoint_interval <= 0: checkpoint_interval = self.actual_total_training_steps

        is_last_step_of_training = (self.total_optimizer_steps == self.actual_total_training_steps)

        if ((self.total_optimizer_steps > 0
             and self.total_optimizer_steps % checkpoint_interval == 0)
                or is_last_step_of_training):

            self.checkpoint_counter += 1
            checkpoint_name = f"step_{self.total_optimizer_steps}_epoch_{epoch}"
            if is_last_step_of_training:
                checkpoint_name = f"final_step_{self.total_optimizer_steps}_epoch_{epoch}"

            save_path = Path(self.config.checkpoint_path) / self.sae_model.get_name()
            save_path.mkdir(parents=True, exist_ok=True)
            full_checkpoint_path = save_path / checkpoint_name

            checkpoint_data = {
                'model_state_dict': self.sae_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'l1_scheduler_state_dict': self.l1_scheduler.state_dict(),
                'trainer_total_optimizer_steps': self.total_optimizer_steps,
                'trainer_n_forward_passes_since_fired': self.n_forward_passes_since_fired,
                'trainer_act_freq_scores': self.act_freq_scores,
                'trainer_n_seen_tokens_for_sparsity': self.n_seen_tokens_for_sparsity,
                'config': self.config.to_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint_data, str(full_checkpoint_path) + ".pt")
            self.checkpoint_save_paths.append(full_checkpoint_path)
            print(f"Saved checkpoint: {full_checkpoint_path}.pt")

            if self.config.log_to_wandb and wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"{self.sae_model.get_name()}-checkpoint-{self.total_optimizer_steps}",
                    type="model-checkpoint",
                    metadata={
                        "step": self.total_optimizer_steps,
                        "epoch": epoch,
                        "model_name": self.sae_model.get_name(),
                        "config": self.config.to_dict()
                    }
                )
                # artifact.add_file(str(full_checkpoint_path) + ".pt")
                wandb.log_artifact(artifact, aliases=[f"step_{self.total_optimizer_steps}",
                                                      "latest" if is_last_step_of_training \
                                                          else f"ckpt_{self.checkpoint_counter}"])

    def _run_evaluation(self):
        if self.evaluator is None:
            return

        eval_metrics = self.evaluator.evaluate()
        if self.verbose:
            print(f"\nRunning evaluation at step {self.total_optimizer_steps}...")
            print("Evaluation metrics:")
            for k, v in eval_metrics.items():
                print(f"  {k}: {v:.4f}")

        if self.config.log_to_wandb and wandb.run:
            # Prefix metrics with "eval/" for W&B
            wandb_eval_metrics = {f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items()}
            wandb.log(wandb_eval_metrics, step=self.total_optimizer_steps)

        self.sae_model.train()

    def _resample_dead_neurons(self, batch_activations: torch.Tensor):
        with torch.no_grad():
            dead_indices = torch.where(self.dead_neuron_mask)[0]
            if len(dead_indices) == 0:
                return

            print(f"\nResampling {len(dead_indices)} dead neurons.")

            # Forward pass to get reconstruction error for the current batch
            step_output = self.sae_model.training_forward_pass(batch_activations, store_activations=True)
            sae_in = step_output.sae_in.reshape(-1, self.config.d_in) # Flatten batch and patch dims
            sae_out = step_output.sae_out.reshape(-1, self.config.d_in)

            # Calculate loss per token
            losses = torch.sum((sae_in - sae_out) ** 2, dim=-1)

            # Use squared loss as probability distribution for sampling
            probs = losses / torch.sum(losses)

            # Sample indices from the batch based on the loss
            replacement_indices = torch.multinomial(probs, len(dead_indices), replacement=True)

            replacement_vectors = sae_in[replacement_indices].to(self.device)
            replacement_vectors_normalized = replacement_vectors / (torch.norm(replacement_vectors, dim=-1, keepdim=True) + 1e-8)

            for i, neuron_idx in enumerate(dead_indices):
                # Re-initialize weights
                self.sae_model.W_dec.data[neuron_idx, :] = replacement_vectors_normalized[i]
                self.sae_model.W_enc.data[:, neuron_idx] = replacement_vectors_normalized[i]
                self.sae_model.b_enc.data[neuron_idx] = 0.0

                # Reset optimizer states for the specific parameters
                for p_group in self.optimizer.param_groups:
                    for p in p_group['params']:
                        if p.grad is not None:
                            # For W_enc
                            if p is self.sae_model.W_enc:
                                self.optimizer.state[p]['exp_avg'][:, neuron_idx] = 0
                                self.optimizer.state[p]['exp_avg_sq'][:, neuron_idx] = 0
                            # For W_dec
                            elif p is self.sae_model.W_dec:
                                self.optimizer.state[p]['exp_avg'][neuron_idx] = 0
                                self.optimizer.state[p]['exp_avg_sq'][neuron_idx] = 0
                            # For b_enc
                            elif p is self.sae_model.b_enc:
                                self.optimizer.state[p]['exp_avg'][neuron_idx] = 0
                                self.optimizer.state[p]['exp_avg_sq'][neuron_idx] = 0

            # Reset the dead neuron tracker for the resampled neurons
            self.n_forward_passes_since_fired[dead_indices] = 0

    def train(self, num_epochs: int = 1):
        if num_epochs <= 0: raise ValueError("Number of epochs must be positive.")
        self.sae_model.train()

        print(f"Starting training on device: {self.device}")
        print(f"Config: Arch={self.config.architecture}, L1_final={self.config.l1_coefficient}, "
              f"LR={self.config.lr}, BatchSize={self.config.train_batch_size}, "
              f"Total Optimizer Steps Target: {self.actual_total_training_steps}")

        if self.config.log_to_wandb and wandb.run:
            wandb.config.update(
                {"derived_total_training_steps": self.actual_total_training_steps, "num_epochs_over_data": num_epochs}
            )

        for epoch in range(1, num_epochs + 1):
            if self.total_optimizer_steps >= self.actual_total_training_steps:
                print(f"Target total training steps ({self.actual_total_training_steps}) reached. Stopping training.")
                break

            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            batch_iterator = iter(self.activation_store)
            steps_per_epoch = len(self.activation_store)
            progress_bar = tqdm(
                batch_iterator, total=steps_per_epoch, desc=f"Epoch {epoch}"
            )

            for batch_idx, batch_activations in enumerate(progress_bar):
                if self.total_optimizer_steps >= self.actual_total_training_steps:
                    break

                current_batch_size = batch_activations.shape[0]
                current_dead_neuron_mask = self.dead_neuron_mask
                current_l1_for_loss = self._get_current_l1_coeff_for_loss()

                if self.config.use_decoder_unit_norm:
                    self.sae_model.set_decoder_norm_to_unit_norm()

                with torch.amp.autocast("cuda",
                                        dtype=self.autocast_dtype,
                                        enabled=self.grad_scaler.is_enabled()):
                    step_output: TrainStepOutput = self.sae_model.training_forward_pass(
                        batch_activations,
                        current_l1_coefficient=current_l1_for_loss,
                        dead_neuron_mask=current_dead_neuron_mask,
                        store_activations=True
                    )
                    total_loss = step_output.total_loss

                self.optimizer.zero_grad()
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)

                if self.config.use_decoder_unit_norm:
                    self.sae_model.remove_gradient_parallel_to_decoder_directions()

                if self.config.clip_gradients:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.sae_model.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                self.total_optimizer_steps += 1

                if (self.config.dead_feature_resampling_interval is not None and
                        self.total_optimizer_steps % self.config.dead_feature_resampling_interval == 0):
                    print("Resampling neurons if necessary...")
                    self._resample_dead_neurons(batch_activations)

                if self.lr_scheduler.optimizer == self.optimizer:
                    self.lr_scheduler.step()
                self.l1_scheduler.step()

                if step_output.feature_acts is not None:
                    self._update_dead_neuron_stats(step_output.feature_acts, current_batch_size)

                if self.total_optimizer_steps % self.config.wandb_log_frequency_steps == 0 or \
                        batch_idx == steps_per_epoch - 1:
                    log_data = step_output.to_dict()
                    num_dead = current_dead_neuron_mask.sum().item()
                    log_data["epoch"] = epoch
                    log_data["batch_in_epoch"] = batch_idx + 1
                    log_data["dead_neurons"] = float(num_dead)
                    log_data["learning_rate"] = self.optimizer.param_groups[0]['lr']
                    log_data["current_l1_coeff"] = current_l1_for_loss

                    progress_bar.set_postfix(
                        {k: f"{v:.4e}" if isinstance(v, float) and abs(v) < 1e-3 or abs(v) > 1e4 else v for k, v in
                         log_data.items() if k not in ["epoch", "batch_in_epoch"]})
                    if self.config.log_to_wandb and wandb.run:
                        wandb.log(log_data, step=self.total_optimizer_steps)

                if self.total_optimizer_steps > 0 and \
                        self.config.eval_interval_steps > 0 and \
                        self.total_optimizer_steps % self.config.eval_interval_steps == 0:
                    self._run_evaluation()

                self._checkpoint_if_needed(epoch)

        if self.evaluator and \
                (self.config.eval_interval_steps == 0 or
                 self.total_optimizer_steps % self.config.eval_interval_steps != 0):
            print("\nRunning final evaluation...")
            self._run_evaluation()

        print("\nTraining finished.")
        if self.config.log_to_wandb and wandb.run:
            if not (self.total_optimizer_steps == self.actual_total_training_steps and self.config.n_checkpoints > 0):
                self._checkpoint_if_needed(num_epochs)

            if self.config.n_checkpoints > 0 and self.checkpoint_save_paths:
                final_model_path = self.checkpoint_save_paths[-1]
                if Path(str(final_model_path) + ".pt").exists():
                    artifact = wandb.Artifact(
                        name=f"{self.sae_model.get_name()}-final_model", type="model",
                        metadata={"step": self.total_optimizer_steps, "epoch": num_epochs,
                                  "config": self.config.to_dict()}
                    )
                    # artifact.add_file(str(final_model_path) + ".pt")
                    wandb.log_artifact(artifact, aliases=["final", "latest_trained"])
            wandb.finish()