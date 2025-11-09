from src.backbones.vit import ViT, ViTOutput
import torch.nn as nn
import timm
import torch

class Dinov2(ViT, nn.Module):

    _model_type: str = "dinov2"
    _embedding_dim: int = 768
    _model_name: str = 'vit_base_patch14_reg4_dinov2.lvd142m'
    _patch_size: int = 14

    def __init__(self, model_name: str | None = None):
        super().__init__()
        self._model_name = model_name if model_name else self._model_name
        self.model = timm.create_model(model_name=self.checkpoint, pretrained=True, num_classes=0)

    def forward(self, x: torch.Tensor) -> ViTOutput:
        x = self.model.forward_features(x)

        cls_token = x[:, 0]

        patch_start_index = self.model.num_prefix_tokens
        patch_tokens = x[:, patch_start_index:]

        return ViTOutput(
            cls_embedding=cls_token,
            patch_embeddings=patch_tokens
        )
    
    @property
    def layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

class Dinov3(ViT, nn.Module):

    _model_type: str = "dinov3"
    _embedding_dim: int = 768
    _patch_size: int = 16
    _model_name: str = 'vit_base_patch16_dinov3.lvd1689m'

    def __init__(self, model_name: str | None = None, checkpoint_path: str | None = None):
        super().__init__()
        self._model_name = model_name if model_name else self._model_name
        self.model = timm.create_model(model_name=self.checkpoint, pretrained=True, num_classes=0, checkpoint_path=checkpoint_path)

    def forward(self, x: torch.Tensor) -> ViTOutput:
        x = self.model.forward_features(x)

        cls_token = x[:, 0]

        patch_start_index = self.model.num_prefix_tokens
        patch_tokens = x[:, patch_start_index:]

        return ViTOutput(
            cls_embedding=cls_token,
            patch_embeddings=patch_tokens
        )


    @property
    def layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

if __name__ == "__main__":
    mod = Dinov2()
    print(mod.checkpoint)
    print(mod.name)
    print(mod.embedding_dim)
    print(mod.patch_size)
    print(mod.num_registers)

    dummy = torch.randn(1,3,518,518)
    out = mod(dummy)

    print(out.cls_embedding.shape)
    print(out.patch_embeddings.shape)
