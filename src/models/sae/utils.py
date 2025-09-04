import json
from pathlib import Path
from typing import Dict, Literal

import yaml
from safetensors.torch import save_file, load_file

import torch
from tqdm import tqdm

from src.models.sae import VisionActivationStore
from src.models.sae.architectures import StandardSAE, TopKSAE
from src.models.sae.config import load_config_from_dict

###################################################################################
################################ CONSTANTS ########################################
###################################################################################

DTYPE_MAP = {
    "torch.float32": torch.float32,
    "float32": torch.float32,
    "torch.float16": torch.float16,
    "float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"

SAE_MAP = {
    "standard": StandardSAE,
    "topk": TopKSAE,
}

###################################################################################
############################# LOADING & SAVING ####################################
###################################################################################


def save_sae(sae_instance: 'SAE', dir_path: str | Path):
    """
    Saves the SAE's state_dict and configuration to the specified directory.

    Args:
        sae_instance: The SAE model instance to save.
        dir_path: The directory path where the model and config will be saved.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = dir_path / SAE_WEIGHTS_FILENAME
    save_file(sae_instance.state_dict(), weights_path)
    print(f"SAE weights saved to: {weights_path}")

    # Save configuration
    config_path = dir_path / SAE_CFG_FILENAME
    config_dict = sae_instance.cfg.to_dict()
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"SAE configuration saved to: {config_path}")


def load_sae(
        dir_path: str | Path,
        device: str | torch.device | None = None,
) -> 'SAE':
    """
    Loads an SAE model and its configuration from the specified directory.

    Args:
        dir_path: The directory path from where to load the model and config.
        device: The device to load the model onto. If None, uses the device
                specified in the config or defaults to CPU.

    Returns:
        The loaded SAE model instance.
    """
    dir_path = Path(dir_path)

    # Load configuration
    config_path = dir_path / SAE_CFG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    cfg = load_config_from_dict(config_dict)

    if device is None:
        device_str = "cpu"
        resolved_device = torch.device(device_str)
    else:
        resolved_device = torch.device(device)

    sae_architecture = cfg.architecture
    if sae_architecture not in SAE_MAP:
        raise ValueError(f"Unknown SAE architecture '{sae_architecture}' in config. "
                         f"Available architectures: {list(SAE_MAP.keys())}")

    SAEClass = SAE_MAP[sae_architecture]
    loaded_sae = SAEClass(cfg=cfg)

    weights_path = dir_path / SAE_WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict = load_file(weights_path, device="cpu")  # Load to CPU first
    loaded_sae.load_state_dict(state_dict)
    loaded_sae.to(resolved_device)  # Then move the entire model

    print(f"SAE loaded from: {dir_path} onto device: {resolved_device}")
    return loaded_sae

def load_configs_from_yaml(yaml_path: str) -> Dict[str, dict]:
    with open(yaml_path, "r") as f:
        full_cfg = yaml.safe_load(f)

    sae_cfg = full_cfg.get("sae", {})
    act_cfg = full_cfg.get("activation_store", {})
    feat_cfg = full_cfg.get("feature_extractor", {})

    return {
        "sae": sae_cfg,
        "activation_store": act_cfg,
        "feature_extractor": feat_cfg
    }


def compute_geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """
    Computes the geometric median of a set of points using Weiszfeld's algorithm.
    """
    median = torch.mean(points, dim=0)
    for _ in range(max_iter):
        prev_median = median.clone()
        distances = torch.norm(points - median, dim=1)

        # Avoid division by zero for points that are at the current median
        inv_distances = 1.0 / distances
        inv_distances[distances == 0] = 0

        weights = inv_distances / torch.sum(inv_distances)
        median = torch.sum(points * weights.unsqueeze(1), dim=0)

        if torch.norm(median - prev_median) < tol:
            break

    return median

def get_data_center(
        activation_store: VisionActivationStore,
        method: Literal["zeros", "mean", "geometric_median"]
) -> torch.Tensor:
    """
    Computes the center of the dataset in the activation store.
    """
    print(f"Computing data center using method: {method}")

    # Concatenate all patches into a single tensor
    all_patches = torch.cat(
        [batch.reshape(-1, activation_store.d_in) for batch in tqdm(activation_store, desc="Loading data for centering")],
        dim=0
    )

    if method == "mean":
        return torch.mean(all_patches, dim=0)
    elif method == "geometric_median":
        return compute_geometric_median(all_patches)
    else:
        raise ValueError(f"Unknown data centering method: {method}")