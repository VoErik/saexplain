from src.utils.get_time import ExecTimer
from src.utils.load_config import load_config, merge_configs
from src.utils.hooks import HookedModel
from src.registry import register_backbone_class
from src.backbones.dino import Dinov2, Dinov3
from src.backbones.mae import MAE
from src.backbones.clip import CLIP

register_backbone_class("dinov2", Dinov2)
register_backbone_class("dinov3", Dinov3)
register_backbone_class("mae", MAE)
register_backbone_class("clip", CLIP)

__all__ = ["ExecTimer", "load_config", "merge_configs"]