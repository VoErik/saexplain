from pathlib import Path

import torch

def load_encoder(name: str, path: str, device: str, **kwargs):
    assert any(arch in name.lower() for arch in ["clip", "dinov2", "dinov3", "mae"]), \
        f"Architecture {name} not recognized."

    if "clip" in name.lower():
        """
        Loads a pretrained OpenCLIP model and its image preprocessor.
        """
        try:
            import open_clip
            arch = kwargs.get("clip_architecture", "ViT-L-14")
            pretrained = kwargs.get("pretrained", "datacomp_xl_s13b_b90k")
            print(f"Loading OpenCLIP model: {arch} pretrained on {pretrained}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch,
                pretrained=pretrained,
                device=device
            )
            tokenizer = open_clip.get_tokenizer(arch)
            print("CLIP model loaded successfully.")
            return model, preprocess
        except ImportError as e:
            print(e)

    elif "dinov2" in name.lower():
        """
        Loads a pretrained DINOv2 model and its image preprocessor.
        """
        try:
            from torchvision import transforms
            print(f"Loading DINOv2 model: {name}")
            model = torch.hub.load('facebookresearch/dinov2', name)
            model = model.to(device)

            preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            print("DINOv2 model loaded successfully.")
            return model, preprocess
        except ImportError as e:
            print(e)

    elif "dinov3" in name.lower():
        """
        Loads a pretrained DINOv3 model and its image preprocessor.
        """
        try:
            from torchvision import transforms

            print(f"Loading DINOv3 model: {name}")
            REPO_DIR = Path(path)
            WEIGHT_PATH = REPO_DIR / "dinov3_weights"/ get_dinov3_weights(name)
            model = torch.hub.load(str(REPO_DIR), name, source='local', weights=str(WEIGHT_PATH))
            model = model.to(device)

            def make_transform(resize_size: int = 224):
                to_tensor = transforms.ToTensor()
                resize = transforms.Resize((resize_size, resize_size), antialias=True)
                normalize = transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
                return transforms.Compose([to_tensor, resize, normalize])

            preprocess = make_transform()

            print("DINOv3 model loaded successfully.")
            return model, preprocess
        except ImportError as e:
            print(e)

    elif "mae" in name.lower():
        return "MAE"

    else:
        raise ValueError(f"Encoder could not be loaded from {path}.")


def get_dinov3_weights(model_name):
    WEIGHTS_DICT = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vit7b16": None
    }
    return WEIGHTS_DICT[model_name]