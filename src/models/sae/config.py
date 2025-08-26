from dataclasses import dataclass, field
from typing import Any, Literal
import torch


@dataclass
class SAEConfig:
    """
    Configuration for a SAE for inference.
    """
    d_in: int
    d_sae: int
    activation_fn_str: str = "relu"
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    apply_b_dec_to_input: bool = True
    embedding_source_name: str | None = None
    architecture: Literal["standard", "topk"] = "standard"


    def __post_init__(self):
        if not isinstance(self.d_in, int) or self.d_in <= 0:
            raise ValueError("d_in must be a positive integer.")
        if not isinstance(self.d_sae, int) or self.d_sae <= 0:
            raise ValueError("d_sae must be a positive integer.")
        if not isinstance(self.activation_fn_str, str):
            raise ValueError("activation_fn_str must be a string.")
        if not isinstance(self.apply_b_dec_to_input, bool):
            raise ValueError("apply_b_dec_to_input must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the configuration to a dictionary."""
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "embedding_source_name": self.embedding_source_name,
            "architecture": self.architecture,
            "config_type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
        """Creates an SAEConfig instance from a dictionary."""
        config_dict_copy = config_dict.copy()
        config_dict_copy.pop("config_type", None)
        return cls(**config_dict_copy)


@dataclass
class TrainingSAEConfig(SAEConfig):
    """
    Configuration for training a SAE, extending the base SAEConfig.
    """
    # Training-specific parameters
    l1_coefficient: float = 0.001
    initial_l1_coefficient: float = 0.0
    l1_warm_up_steps: int = 1000
    lr: float = 1e-3
    lr_end_factor: float = 0.0 # 0.1 would be 10% of initial lr
    lr_scheduler_name: str | None = "cosine_warmup_restarts"
    lr_warm_up_steps: int = 500
    # lr_decay_steps: int | None = 10000 # not used currently
    train_batch_size: int = 32
    total_training_steps: int | None = 30000
    clip_gradients: bool = False

    dead_feature_window: int = 10000 # convention for LLM SAEs -> change to smaller val probably
    # feature_sampling_window: int = 2000 # How often to log/reset sparsity stats (for later)

    # TopK specific parameters
    topk_aux_loss_coefficient: float = 1.0
    aux_k: int | None = None # Number of top dead features to use in aux loss. If None, use d_sae // 2 or similar.

    # Checkpointing
    n_checkpoints: int = 5
    checkpoint_path: str = "./experiments/sae_checkpoints"

    # Evaluation parameters
    eval_interval_steps: int = 1000
    eval_n_batches: int | None = None
    eval_batch_size: int = 128

    # W&B Logging
    log_to_wandb: bool = True
    wandb_dir: str | None = None
    wandb_project: str = "thesis"
    wandb_entity: str | None = None
    wandb_log_frequency_steps: int = 500  # Log every N steps
    wandb_run_name: str | None = None

    # MP
    autocast_fp16: bool = False
    autocast_bf16: bool = True

    def __post_init__(self):
        super().__post_init__()
        if (self.l1_coefficient < 0) or (not isinstance(self.l1_coefficient, float)):
            raise ValueError("l1_coefficient must be a positive float or zero.")
        if not isinstance(self.initial_l1_coefficient, float) or self.initial_l1_coefficient < 0:
            raise ValueError("initial_l1_coefficient must be non-negative.")
        if not isinstance(self.l1_warm_up_steps, int) or self.l1_warm_up_steps < 0:
            raise ValueError("l1_warm_up_steps must be a non-negative integer.")

        if (self.lr <= 0) or (not isinstance(self.lr, float)):
            raise ValueError("lr must be a positive float.")
        if (self.lr_end_factor < 0) or (not isinstance(self.lr_end_factor, float)):
            raise ValueError("lr_end_factor must be a positive float or zero.")
        if self.lr_scheduler_name is not None and not isinstance(self.lr_scheduler_name, str):
            raise ValueError("lr_scheduler_name must be a string or None.")
        if not isinstance(self.lr_warm_up_steps, int) or self.lr_warm_up_steps < 0:
            raise ValueError("lr_warm_up_steps must be a non-negative integer.")

        if self.total_training_steps is not None and (
                not isinstance(self.total_training_steps, int) or self.total_training_steps <= 0):
            raise ValueError("total_training_steps must be a positive integer if specified.")
        if (self.train_batch_size <= 0) or (not isinstance(self.train_batch_size, int)):
            raise ValueError("train_batch_size must be a positive integer.")

        if (self.dead_feature_window <= 0) or (not isinstance(self.dead_feature_window, int)):
            raise ValueError("dead_feature_window must be a positive integer.")

        if self.n_checkpoints < 0: raise ValueError("n_checkpoints must be non-negative.")

        if not isinstance(self.wandb_log_frequency_steps, int) or self.wandb_log_frequency_steps <= 0:
            raise ValueError("wandb_log_frequency_steps must be positive.")

        if self.autocast_fp16 and self.autocast_bf16:
            print("Warning: Both autocast_fp16 and autocast_bf16 are True. bf16 will be preferred if available.")

        if not isinstance(self.eval_interval_steps, int) or self.eval_interval_steps <= 0:
            raise ValueError("eval_interval_steps must be a positive integer.")
        if self.eval_n_batches is not None and (not isinstance(self.eval_n_batches, int) or self.eval_n_batches <= 0):
            raise ValueError("eval_n_batches must be a positive integer or None.")
        if not isinstance(self.eval_batch_size, int) or self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "l1_coefficient": self.l1_coefficient,
            "initial_l1_coefficient": self.initial_l1_coefficient,
            "l1_warm_up_steps": self.l1_warm_up_steps,
            "lr": self.lr,
            "lr_scheduler_name": self.lr_scheduler_name,
            "lr_warm_up_steps": self.lr_warm_up_steps,
            "lr_end_factor": self.lr_end_factor,
            "train_batch_size": self.train_batch_size,
            "dead_feature_window": self.dead_feature_window,
            "total_training_steps": self.total_training_steps,
            "clip_gradients": self.clip_gradients,
            "topk_aux_loss_coefficient": self.topk_aux_loss_coefficient,
            "aux_k": self.aux_k,
            "n_checkpoints": self.n_checkpoints,
            "checkpoint_path": self.checkpoint_path,
            "log_to_wandb": self.log_to_wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "wandb_log_frequency_steps": self.wandb_log_frequency_steps,
            "wandb_run_name": self.wandb_run_name,
            "autocast_fp16": self.autocast_fp16,
            "autocast_bf16": self.autocast_bf16,
            "eval_interval_steps": self.eval_interval_steps,
            "eval_n_batches": self.eval_n_batches,
            "eval_batch_size": self.eval_batch_size,
            "config_type": self.__class__.__name__,
        })
        return data

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        """Creates a TrainingSAEConfig instance from a dictionary."""
        config_dict_copy = config_dict.copy()
        config_dict_copy.pop("config_type", None)
        return cls(**config_dict_copy)

# Helper to load the correct config type
def load_config_from_dict(config_dict: dict[str, Any]) -> SAEConfig | TrainingSAEConfig:
    config_type = config_dict.get("config_type")
    if config_type == "TrainingSAEConfig":
        return TrainingSAEConfig.from_dict(config_dict)
    elif config_type == "SAEConfig":
        return SAEConfig.from_dict(config_dict)
    else:
        print(f"Warning: 'config_type' field missing or unknown ('{config_type}'). Attempting to load as SAEConfig.")
        return SAEConfig.from_dict(config_dict)