# Basically taken from 
# @misc{bloom2024saetrainingcodebase,
#   title = {SAELens},
#   author = {Bloom, Joseph and Tigges, Curt and Duong, Anthony and Chanin, David},
#   year = {2024},
#   howpublished = {\url{https://github.com/jbloomAus/SAELens}},
# }
# with miniscule adjustments and simplified.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, asdict
from typing import NamedTuple, Any, TypeVar, Generic, Literal, Callable
from typing_extensions import override

import einops
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from numpy.typing import NDArray
from pathlib import Path
from safetensors.torch import save_file, load_file

from sae.utils.misc import DTYPE_MAP, filter_valid_dataclass_fields, SAE_CFG_FILENAME, SAE_WEIGHTS_FILENAME
from sae.utils.sbp import calculate_sbp_loss
from sae.registry import get_sae_class, get_sae_training_class

T = TypeVar("T")

@dataclass
class SAEConfig(ABC):
    d_in: int = 768   # Dimension of the SAE input
    d_sae: int = 1536 # Hidden SAE dimension
    device: str = "cpu"
    dtype: str = "float32"
    apply_b_dec_to_input: bool = True
    normalize_activations: Literal[
        "none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"
    ] = "none"

    @classmethod
    @abstractmethod
    def architecture(cls) -> str: ...


    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        cfg_class = get_sae_class(config_dict["architecture"])[1]
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cfg_class)
        res = cfg_class(**filtered_config_dict)
        if not isinstance(res, cls):
            raise ValueError(
                f"SAE config class {cls} does not match dict config class {type(res)}"
            )
        return res
    

    def to_dict(self) -> dict[str, Any]:
        res = {field.name: getattr(self, field.name) for field in fields(self)}
        res["architecture"] = self.architecture()
        return res

@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig, ABC):
    # https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    # 0.1 corresponds to the "heuristic" initialization, use None to disable
    decoder_init_norm: float | None = 0.1
    apply_sbp: bool = False
    ema_beta: float | None = 0.99
    num_classes: int | None = None
    sbp_lambda: float = 10.0
    sbp_alpha: float = 0.5
    sbp_threshold: float = 1e-5
    sbp_method: str = "entropy"

    @classmethod
    @abstractmethod
    def architecture(cls) -> str: ...


    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any]
    ):
        cfg_class = cls
        if "architecture" in config_dict:
            cfg_class = get_sae_training_class(config_dict["architecture"])[1]
        if not issubclass(cfg_class, cls):
            raise ValueError(
                f"SAE config class {cls} does not match dict config class {type(cfg_class)}"
            )

        valid_config_dict = filter_valid_dataclass_fields(config_dict, cfg_class)
        return cfg_class(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            **asdict(self),
            "architecture": self.architecture(),
        }
    
    def get_inference_config_class(self) -> type[SAEConfig]:
        """
        Get the architecture for inference.
        """
        return get_sae_class(self.architecture())[1]

    def get_inference_sae_cfg_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing attributes corresponding to the fields
        defined in the base SAEConfig class.
        """
        base_sae_cfg_class = self.get_inference_config_class()
        base_config_field_names = {f.name for f in fields(base_sae_cfg_class)}
        result_dict = {
            field_name: getattr(self, field_name)
            for field_name in base_config_field_names
        }
        result_dict["architecture"] = base_sae_cfg_class.architecture()
        return result_dict

@dataclass
class TrainStepInput:
    sae_in: torch.Tensor
    coefficients: dict[str, float]
    dead_neuron_mask: torch.Tensor | None
    n_training_steps: int
    labels: torch.Tensor | None = None

@dataclass 
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor | float | int] = field(default_factory=dict)

class TrainCoefficientConfig(NamedTuple):
    value: float
    warm_up_steps: int


class SAE(nn.Module, ABC, Generic[T]):

    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = DTYPE_MAP[cfg.dtype]

        self.initialize_weights()
        self.activation_fn = self.get_activation_fn()
        self._setup_activation_normalization()


    def initialize_weights(self):
        """Initialize model weights."""
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

        w_enc_data = self.W_dec.data.T.clone().detach().contiguous()
        self.W_enc = nn.Parameter(w_enc_data)

    
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.ReLU()
    
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        self.W_enc.data *= scaling_factor  # type: ignore
        self.W_dec.data /= scaling_factor  # type: ignore
        self.b_dec.data /= scaling_factor  # type: ignore
        self.cfg.normalize_activations = "none"

    def _setup_activation_normalization(self):
        """Set up activation normalization functions based on config."""
        if self.cfg.normalize_activations == "constant_norm_rescale":

            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                return x * self.x_norm_coeff

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:
                x = x / self.x_norm_coeff  # type: ignore
                del self.x_norm_coeff
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out
        elif self.cfg.normalize_activations == "layer_norm":
            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(
                x: torch.Tensor,
                eps: float = 1e-5,  # noqa: ARG001
            ) -> torch.Tensor:
                return x * self.ln_std + self.ln_mu  # type: ignore

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Only update b_enc if it exists (standard/jumprelu architectures)
        if hasattr(self, "b_enc") and isinstance(self.b_enc, nn.Parameter):
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        For inference, just encode without returning hidden_pre.
        (training_forward_pass calls encode_with_hidden_pre).
        """
        ...

    def decode(
        self, features: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        ...

    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:

        sae_in = sae_in.to(self.dtype)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        bias_term = self.b_dec * self.cfg.apply_b_dec_to_input

        return sae_in - bias_term

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def get_name(self):
        """Generate a name for this SAE."""
        return f"sae_{self.cfg.architecture()}_d{self.cfg.d_sae}"

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Save model weights and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Generate the weights
        state_dict = self.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.to_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        return model_weights_path, cfg_path

    @classmethod
    def load(cls, dir_path: str | Path):
        """
        Loads SAE from directory.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise ValueError(f"{dir_path} does not exist.")
        
        config_path = dir_path / SAE_CFG_FILENAME
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        sae_config_cls = cls.get_sae_config_class_for_architecture(
            config_dict["architecture"]
        )
        sae_cfg = sae_config_cls.from_dict(config_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture())
        sae = sae_cls(sae_cfg)

        weights_path = dir_path / SAE_WEIGHTS_FILENAME
        state_dict = load_file(weights_path, device="cpu")
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Create an SAE from a config dictionary."""
        sae_cls = cls.get_sae_class_for_architecture(config_dict["architecture"])
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            config_dict["architecture"]
        )
        return sae_cls(sae_config_cls.from_dict(config_dict))
    
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        pass

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        state_dict.pop("ema_matrix", None)

    @classmethod
    def get_sae_class_for_architecture(
        cls, architecture: str
    ):
        """Get the SAE class for a given architecture."""
        sae_cls, _ = get_sae_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    @classmethod
    def get_sae_config_class_for_architecture(
        cls,
        architecture: str,  # noqa: ARG003
    ) -> type[SAEConfig]:
        return SAEConfig

    
class TrainingSAE(SAE[T], ABC):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mse_loss_fn = mse_loss

        if self.cfg.apply_sbp:
            self.register_buffer(
            "ema_matrix", 
            torch.zeros(self.cfg.d_sae, self.cfg.num_classes)
        )

    @abstractmethod
    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Encode with access to pre-activation values for training."""
        ...

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        For inference, just encode without returning hidden_pre.
        (training_forward_pass calls encode_with_hidden_pre).
        """
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def decode(
        self, features: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        sae_out_pre = einops.einsum(
            features, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
        ) + self.b_dec
        sae_out = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out
    

    @override
    def initialize_weights(self):
        super().initialize_weights()
        if self.cfg.decoder_init_norm is not None:
            with torch.no_grad():
                self.W_dec.data /= self.W_dec.norm(dim=-1, keepdim=True)
                self.W_dec.data *= self.cfg.decoder_init_norm
            self.W_enc.data = self.W_dec.data.T.clone().detach().contiguous()

    @abstractmethod
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""
        ...

    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

        # Apply same bird penalty
        if self.cfg.apply_sbp:
            sbp_loss = self.calculate_sbp(
                step_input=step_input,
                feature_acts=feature_acts
            )
            losses.update({"sbp_loss": sbp_loss})
            total_loss += sbp_loss

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )
    @abstractmethod
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]: 
        ...

    def calculate_sbp(self, step_input: TrainStepInput, feature_acts: torch.Tensor):
        """Calculates the single bird penalty loss term."""
        if step_input.labels is None:
            raise ValueError("Labels are needed to calculate this penalty.")
        labels_onehot = F.one_hot(step_input.labels, num_classes=self.cfg.num_classes).float()
        weights = feature_acts.abs().pow(1.0)
        counts = labels_onehot.sum(dim=0, keepdim=True) + 1e-8
        activation_matrix = torch.matmul(weights.t(), labels_onehot) / counts
        loss_input = (1.0 - self.cfg.ema_beta) * activation_matrix + self.cfg.ema_beta * self.ema_matrix.detach()
        sbp_loss = calculate_sbp_loss(
            activation_matrix=loss_input, 
            num_classes=self.cfg.num_classes, 
            alpha=self.cfg.sbp_alpha, 
            support_threshold=self.cfg.sbp_threshold, 
            method=self.cfg.sbp_method
        )

        with torch.no_grad():
            self.ema_matrix = (1.0 - self.cfg.ema_beta) * activation_matrix.detach() + self.cfg.ema_beta * self.ema_matrix

        scaled_sbp_loss = self.cfg.sbp_lambda * sbp_loss
        return scaled_sbp_loss

    @torch.no_grad()
    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        W_dec_norm_dist = self.W_dec.detach().float().norm(dim=1).cpu().numpy()
        return {
            "weights/W_dec_norms": W_dec_norm_dist,
        }
    
    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        """Save inference version of model weights and config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Generate the weights
        state_dict = self.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving_inference(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.get_inference_sae_cfg_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        return model_weights_path, cfg_path

    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        """
        Process the state dict for saving the inference model.
        This is a hook that can be overridden to change how the state dict is processed for the inference model.
        """
        return self.process_state_dict_for_saving(state_dict)
    
    @classmethod
    def get_sae_class_for_architecture(
        cls, architecture: str
    ):
        """Get the SAE class for a given architecture."""
        sae_cls, _ = get_sae_training_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    @classmethod
    def get_sae_config_class_for_architecture(
        cls,
        architecture: str,  # noqa: ARG003
    ) -> type[TrainingSAEConfig]:
        return get_sae_training_class(architecture)[1]
    
def mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(preds, target, reduction="none")