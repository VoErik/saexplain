import torch
import torch.nn as nn
import timm
from src.backbones.vit import ViT, ViTOutput

class MAE(ViT, nn.Module):

    _model_type: str = "mae"
    _embedding_dim: int = 768
    _model_name: str = 'vit_base_patch16_224.mae'
    _patch_size: int = 16

    def __init__(self, model_name: str | None = None, checkpoint_path: str | None = None):
        super().__init__()
        self._model_name = model_name if model_name else self._model_name
        self.model = timm.create_model(model_name=self.checkpoint, pretrained=True, num_classes=0, checkpoint_path=checkpoint_path)

    def forward(self, x: torch.Tensor) -> ViTOutput:
        x = self.model.forward_features(x)
        return ViTOutput(
            cls_embedding=x[:, 0],
            patch_embeddings=x[:, 1:]
        )

    @property
    def layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

if __name__ == "__main__":
    mod = MAE()
    print(mod.checkpoint)
    print(mod.name)
    print(mod.embedding_dim)
    print(mod.patch_size)

    dummy = torch.randn(1,3,224,224)
    out = mod(dummy)

    print(out.cls_embedding.shape)
    print(out.patch_embeddings.shape)