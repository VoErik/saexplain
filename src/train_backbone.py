

def train_backbone(cfg):
    """
    Train a backbone model (e.g., CLIP) based on the provided configuration.
    
    Args:
        cfg (dict): Configuration for training.
    """
    
    backbone_type = cfg.get("backbone_type", "clip")
    if backbone_type == "clip":
        from src.backbones.clip import train_clip, run_training_classification
        if cfg.get("mode", None) == "backbone":
            train_clip(cfg)
        else:
            run_training_classification(cfg, num_classes=200)
    elif backbone_type == "mae":
        from src.backbones.mae import train_mae
        train_mae(cfg)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")