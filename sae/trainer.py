import contextlib
import itertools

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Literal
from pathlib import Path

import torch
import wandb
from torch.optim import Adam
from tqdm.auto import tqdm

from sae.utils.scaler import ActivationScaler

from sae.utils.schedulers import (
    CoefficientScheduler,
    get_lr_scheduler
)

from sae.core import (
    TrainCoefficientConfig,
    TrainStepInput,
    TrainStepOutput,
)

from sae.utils.misc import SAE_SPARSITY_FILENAME, filter_valid_dataclass_fields


@dataclass
class LoggingConfig:
    # WANDB
    log_to_wandb: bool = True
    log_model_artifacts_to_wandb: bool = False
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    wandb_project: str = "sae_training"
    wandb_id: str | None = None
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 100 steps.

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res

    def log(
        self,
        trainer: Any,
        weights_path: Path | str,
        cfg_path: Path | str,
        sparsity_path: Path | str | None,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        
        if not self.log_to_wandb or not self.log_model_artifacts_to_wandb:
            print("Skipping artifact logging to W&B.")
            return

        sae_name = trainer.sae.get_name().replace("/", "__")

        # save model weights and cfg
        model_artifact = wandb.Artifact(
            sae_name,
            type="model",
            metadata=dict(trainer.cfg.__dict__),
        )
        model_artifact.add_file(str(weights_path))
        model_artifact.add_file(str(cfg_path))
        wandb.log_artifact(model_artifact, aliases=wandb_aliases)

        # save log feature sparsity
        sparsity_artifact = wandb.Artifact(
            f"{sae_name}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(trainer.cfg.__dict__),
        )
        if sparsity_path is not None:
            sparsity_artifact.add_file(str(sparsity_path))
        wandb.log_artifact(sparsity_artifact)


@dataclass
class SAETrainerConfig:
    total_training_samples: int = 100_000
    device: str = "cuda"
    autocast: bool = True
    lr: float = 5e-5
    lr_end: float | None = 5e-6
    lr_scheduler_name: str = "constant"
    lr_warm_up_steps: int = 1000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1
    train_batch_size_samples: int = 256
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000
    logger: LoggingConfig = field(default_factory=LoggingConfig)
    eval_metric_mode: str = "min"
    eval_metric_to_track: str = "losses/eval_loss"
    model_save_path: str = "./ckpts/sae/"

    # experimental
    apply_sbp: bool = False

    @property
    def total_training_steps(self) -> int:
        return self.total_training_samples // self.train_batch_size_samples
    
    def to_dict(self) -> dict[str, Any]:
        res = {field.name: getattr(self, field.name) for field in fields(self)}
        return res
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()

def _unwrap_item(item: float | torch.Tensor) -> float:
    return item.item() if isinstance(item, torch.Tensor) else item

class SAETrainer:
    """
    SAE Trainer class.
    """

    data_provider: Any
    activation_scaler: ActivationScaler
    evaluator: Any | None
    coefficient_schedulers: dict[str, CoefficientScheduler]

    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae,
        data_provider: Any,
        evaluator: Any | None = None,
    ) -> None:
        self.sae = sae
        # data provider needs to have an attribute "train_dataset" that is of type torch.utils.data.Dataset
        # TODO: make this agnostic, coupling too strong rn
        self.embedding_cache = data_provider 
        self.evaluator = evaluator
        self.activation_scaler = ActivationScaler()
        self.cfg = cfg

        self.n_training_steps: int = 0
        self.n_training_samples: int = 0
        self.started_fine_tuning: bool = False


        if self.cfg.eval_metric_mode not in ["min", "max"]:
            raise ValueError(f"eval_metric_mode must be 'min' or 'max', got {self.cfg.eval_metric_mode}")
        
        self.best_metric = float('inf') if self.cfg.eval_metric_mode == "min" else float('-inf')
        self.best_model_state_dict = None

        self.act_freq_scores = torch.zeros(sae.cfg.d_sae, device=cfg.device)
        self.n_forward_passes_since_fired = torch.zeros(
            sae.cfg.d_sae, device=cfg.device
        )
        self.n_frac_active_samples = 0

        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.coefficient_schedulers = {}
        for name, coeff_cfg in self.sae.get_coefficients().items():
            if not isinstance(coeff_cfg, TrainCoefficientConfig):
                coeff_cfg = TrainCoefficientConfig(value=coeff_cfg, warm_up_steps=0)
            self.coefficient_schedulers[name] = CoefficientScheduler(
                warm_up_steps=coeff_cfg.warm_up_steps,
                final_value=coeff_cfg.value,
            )

        self.grad_scaler = torch.amp.GradScaler(
            device=self.cfg.device, enabled=self.cfg.autocast
        )

        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type=self.cfg.device,
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_samples

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()
    
    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        if (self.n_training_steps + 1) % self.cfg.logger.wandb_log_frequency == 0:
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_samples=self.n_training_samples,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def get_coefficients(self) -> dict[str, float]:
        return {
            name: scheduler.value
            for name, scheduler in self.coefficient_schedulers.items()
        }

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = feature_acts.bool().float().sum(-1).to_dense().mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance_legacy = 1 - per_token_l2_loss / total_variance
        explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
            "metrics/explained_variance_legacy_std": explained_variance_legacy.std().item(),
            "metrics/explained_variance": explained_variance.item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/n_training_samples": n_training_samples,
            **{
                f"details/{name}_coefficient": scheduler.value
                for name, scheduler in self.coefficient_schedulers.items()
            },
        }
        for loss_name, loss_value in output.losses.items():
            log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)

        for metric_name, metric_value in output.metrics.items():
            log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)

        return log_dict
    
    @torch.no_grad()
    def _run_and_log_evals(self) -> dict[str, Any] | None:
        """
        Runs evaluation and logs to W&B.
        Returns the evaluation metrics dictionary.
        """
        eval_metrics = {}
        if (self.n_training_steps + 1) % (
            self.cfg.logger.wandb_log_frequency
            * self.cfg.logger.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            eval_metrics = (
                self.evaluator(self.sae, self.embedding_cache, self.activation_scaler)
                if self.evaluator is not None
                else {}
            )
            for key, value in self.sae.log_histograms().items():
                eval_metrics[key] = wandb.Histogram(value)  # type: ignore

            if self.cfg.logger.log_to_wandb:
                wandb.log(
                    eval_metrics,
                    step=self.n_training_steps,
                )
            self.sae.train()
            return eval_metrics
        
        return None

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        log_feature_sparsity = _log_feature_sparsity(self.feature_sparsity)
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())  # type: ignore
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(
            self.sae.cfg.d_sae,  # type: ignore
            device=self.cfg.device,
        )
        self.n_frac_active_samples = 0


    @torch.no_grad()
    def _update_pbar(
        self,
        step_output: TrainStepOutput,
        pbar: tqdm,  # type: ignore
        update_interval: int = 100,
    ):
        if self.n_training_steps % update_interval == 0:
            loss_strs = " | ".join(
                f"{loss_name}: {_unwrap_item(loss_value):.5f}"
                for loss_name, loss_value in step_output.losses.items()
            )
            pbar.set_description(f"{self.n_training_steps}| {loss_strs}")
            pbar.update(update_interval * self.cfg.train_batch_size_samples)


    def _train_step(
        self,
        sae,
        sae_in: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> TrainStepOutput:
        sae.train()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.logger.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with self.autocast_if_enabled:
            train_step_output = self.sae.training_forward_pass(
                step_input=TrainStepInput(
                    sae_in=sae_in,
                    dead_neuron_mask=self.dead_neurons,
                    coefficients=self.get_coefficients(),
                    n_training_steps=self.n_training_steps,
                    labels=labels
                ),
            )

            with torch.no_grad():
                firing_feats = train_step_output.feature_acts.bool().float()
                did_fire = firing_feats.sum(-2).bool()
                if did_fire.is_sparse:
                    did_fire = did_fire.to_dense()
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += firing_feats.sum(0)
                self.n_frac_active_samples += self.cfg.train_batch_size_samples

        # Grad scaler will rescale gradients if autocast is enabled
        self.grad_scaler.scale(
            train_step_output.loss
        ).backward()  # loss.backward() if not autocasting
        self.grad_scaler.unscale_(self.optimizer)  # needed to clip correctly
        # TODO: Work out if grad norm clipping should be in config / how to test it.
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        self.grad_scaler.step(
            self.optimizer
        )  # just ctx.optimizer.step() if not autocasting
        self.grad_scaler.update()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        for scheduler in self.coefficient_schedulers.values():
            scheduler.step()

        return train_step_output

    def fit(self):

        self.sae.to(self.cfg.device)
        pbar = tqdm(total=self.cfg.total_training_samples, desc="Training SAE")

        train_loader = torch.utils.data.DataLoader(
            self.embedding_cache,
            batch_size=self.cfg.train_batch_size_samples,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        loader_iter = itertools.cycle(train_loader)

        if self.sae.cfg.normalize_activations == "expected_average_only_in":
            est_loader = torch.utils.data.DataLoader(
                self.embedding_cache,
                batch_size=self.cfg.train_batch_size_samples,
                shuffle=True
            )
            self.activation_scaler.estimate_scaling_factor(
                d_in=self.sae.cfg.d_in,
                data_provider=itertools.cycle(est_loader), # Use cycling iterator
                n_batches_for_norm_estimate=int(1e3),
            )

        while self.n_training_samples < self.cfg.total_training_samples:
            # Do a training step.
            batch_raw = next(loader_iter)
            if len(batch_raw) == 2:
                batch, labels = batch_raw
                labels = labels.to(self.sae.device)
            else:
                batch = batch_raw
                labels = None
            batch = batch.to(self.sae.device)
            scaled_batch = self.activation_scaler(batch)
            self.n_training_samples += batch.shape[0]
            
            step_output = self._train_step(sae=self.sae, sae_in=scaled_batch, labels=labels)

            if self.cfg.logger.log_to_wandb:
                self._log_train_step(step_output)
                eval_metrics = self._run_and_log_evals()

                if eval_metrics:
                    if self.cfg.eval_metric_to_track not in eval_metrics:
                        print(f"Warning: Metric '{self.cfg.eval_metric_to_track}' not in eval metrics. Cannot save best model.")
                    else:
                        current_metric = eval_metrics[self.cfg.eval_metric_to_track]
                        
                        is_better = (
                            (self.cfg.eval_metric_mode == "min" and current_metric < self.best_metric) or
                            (self.cfg.eval_metric_mode == "max" and current_metric > self.best_metric)
                        )
                        
                        if is_better:
                            self.best_metric = current_metric
                            self.best_model_state_dict = self.sae.state_dict()
                            print(f"\nNew best model saved with {self.cfg.eval_metric_to_track}: {current_metric:.4f}")

            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

        # fold the estimated norm scaling factor into the sae weights
        if self.activation_scaler.scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activation_scaler.scaling_factor
            )
            self.activation_scaler.scaling_factor = None

        if self.best_model_state_dict is not None:
            # Load the best model weights
            self.sae.load_state_dict(self.best_model_state_dict)
            print(f"Loaded best model (metric: {self.best_metric:.4f}) for final save.")
        else:
            print("Warning: `save_best_model` was True, but no best model was saved. Saving final model.")

        # Create save directory
        save_dir = Path(self.cfg.model_save_path) / self.sae.get_name()
        save_dir.mkdir(parents=True, exist_ok=True)
        

        sparsity_path = save_dir / SAE_SPARSITY_FILENAME

        self.sae.save(save_dir)
        torch.save(self.feature_sparsity, sparsity_path)
        
        print(f"Saved best model and artifacts to {save_dir}")

        

        return self.sae

@dataclass
class TrainingRunnerConfig:
    # TRAINER CONFIG PARAMS
    total_training_samples: int = 100_000
    device: str = "cuda"
    autocast: bool = True
    lr: float = 5e-5
    lr_end: float | None = 5e-6
    lr_scheduler_name: str = "constant"
    lr_warm_up_steps: int = 1000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1
    train_batch_size_samples: int = 256
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000
    logger: LoggingConfig = field(default_factory=LoggingConfig)
    eval_metric_mode: str = "min"
    eval_metric_to_track: str = "losses/eval_loss"
    model_save_path: str = "./ckpts/sae/"

    # EMBEDDING CACHE CONFIG PARAMS
    datasets: list[str] = field(default_factory=lambda: ["fitzpatrick"])
    data_root: str = "../data"
    num_classes: int = 200
    model_path: str = "../ckpts/clip/openai-clip-vit-base-patch16-['ham', 'fitzpatrick', 'scin', 'midas']-best_model"
    cache_dir: str = "../cache"
    cls_only: bool = False
    extraction_batch_size: int = 64
    layer_index: int = -1

    # EVALUATOR CONFIG PARAMS

    # SAE CONFIG PARAMS
    ## Basic configuration
    d_in: int = 768
    d_sae: int = 1536
    device: str = "cuda"
    dtype: str = "float32"
    apply_b_dec_to_input: bool = True
    decoder_init_norm: float | None = 0.1
    normalize_activations: str = "expected_average_only_in"
    architecture: str = "jumprelu"
    apply_sbp: bool = False
    sbp_alpha: float = 0.5
    sbp_threshold: float = 1e-5
    sbp_method: str = "entropy"
    sbp_lambda: float = 20.0
    ema_beta: float = 0.1

    ## relu
    l1_coefficient: float = 3.0
    lp_norm: float = 1.0
    l1_warm_up_steps: int = 0

    ## topk
    k: int = 20
    rescale_acts_by_decoder_norm: bool = False
    aux_loss_coefficient: float = 1.0

    ## batchtopk
    topk_threshold_lr: float = 0.01

    ## matryoshka
    matryoshka_widths: list[int] = field(default_factory=list)

    ## jumprelu
    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "step"
    l0_coefficient: float = 1.0
    l0_warm_up_steps: int = 0
    pre_act_loss_coefficient: float | None = None
    jumprelu_tanh_scale: float = 4.0

    # LOGGING CONFIG PARAMS
    log_to_wandb: bool = True
    log_model_artifacts_to_wandb: bool = False
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    wandb_project: str = "sae_training"
    wandb_id: str | None = None
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 100 steps.

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)
        return res
    
    def __repr__(self) -> str:
        """
        Generates a human-readable, multi-line representation of the config.
        This is much cleaner than the default dataclass repr for large configs.
        """
        # Get all field objects from the dataclass definition
        all_fields = fields(self)
        
        lines = [f"{self.__class__.__name__}("]
        
        # Add each field as '    field_name=value,'
        for f in all_fields:
            field_name = f.name
            value = getattr(self, field_name)
            # Use !r to get the 'repr' of the value itself (e.g., adds quotes for strings)
            lines.append(f"    {field_name}={value!r},")
        
        # Add the closing parenthesis
        lines.append(")")
        
        # Join all lines with a newline
        return "\n".join(lines)


