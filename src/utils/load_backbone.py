

def load_backbone(model_name_or_path: str):
    if "clip" in model_name_or_path.lower():
        from src.backbones.clip import CLIP, CLIPTransform
        model = CLIP(model_name_or_path=model_name_or_path)
        transform = CLIPTransform(model.processor)
        return model, transform
    elif "dino" in model_name_or_path.lower():
        raise NotImplementedError("Dino architecture not supported at the moment.")
    else:
        raise ValueError("Model architecture unknown.")