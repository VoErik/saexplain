def load_backbone(model_name_or_path: str, is_train: bool = False):
    if "clip" in model_name_or_path.lower():
        from src.backbones.clip import CLIP, CLIPTransform
        model = CLIP(model_name_or_path=model_name_or_path)
        transform = CLIPTransform(model.processor, is_train=is_train)
        return model, transform
    elif "dino" in model_name_or_path.lower():
        raise NotImplementedError("Dino architecture not supported at the moment.")
    else:
        raise ValueError("Model architecture unknown.")