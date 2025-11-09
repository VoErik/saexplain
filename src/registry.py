from typing import TYPE_CHECKING, Any

# avoid circular imports
if TYPE_CHECKING:
    from src.backbones.vit import ViT

BACKBONE_CLASS_REGISTRY: dict[str, "type[ViT[Any]]"] = {}

def register_backbone_class(
    architecture: str,
    backbone_class: "type[ViT[Any]]",
) -> None:
    if architecture in BACKBONE_CLASS_REGISTRY:
        raise ValueError(
            f"Backbone class for architecture {architecture} already registered."
        )
    BACKBONE_CLASS_REGISTRY[architecture] = backbone_class

def get_backbone_class(
    architecture: str,
) -> "type[ViT[Any]]":
    return BACKBONE_CLASS_REGISTRY[architecture]
