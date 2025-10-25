from src.utils import ExecTimer, load_config

def run_training(config_path):
    """
    Run the training process based on the provided configuration file.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    
    print(f"Running training with config: {config_path}")

    cfg = load_config(config_path)

    if cfg.get("mode", None) == "backbone":
        from src.backbones import train_backbone

        with ExecTimer("Backbone Training"):
            train_backbone(cfg)

    elif cfg.get("mode", None) == "sae":
        from src.sae import train_sae
        with ExecTimer("SAE Training"):
            train_sae(cfg)

    elif cfg.get("mode", None) == "joint":
        from src.misc import train_joint
        with ExecTimer("Joint Training"):
            train_joint(cfg)
    
    else:
        raise ValueError(f"Unknown training mode: {cfg.get('mode', None)}")