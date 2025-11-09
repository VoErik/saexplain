import timm
from src.registry import get_backbone_class


def load_backbone(architecture: str, model_name: str | None = None, is_train: bool = False, checkpoint_path: str | None = None):
    if architecture in ["clip", "dinov2", "dinov3", "mae"]:
        model_class = get_backbone_class(architecture)
        model = model_class(model_name=model_name, checkpoint_path=checkpoint_path)
    else:
        ValueError("Model architecture unknown.")
    
    transform_config = timm.data.resolve_model_data_config(model)
    train_transform = timm.data.create_transform(**transform_config, is_training=True)
    val_transform = timm.data.create_transform(**transform_config, is_training=False)
    return model, train_transform, val_transform