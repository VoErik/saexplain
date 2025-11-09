from sae.eval.post_hoc_eval import (
    AtlasConfig,
    run_post_hoc_eval

)
from src.utils.load_backbone import load_backbone
from src.data import get_dataloaders

def run_post_hoc_evaluation(configuration: dict):
    cfg = AtlasConfig.from_dict(configuration)

    backbone, _, eval_transform = load_backbone(
        architecture=cfg.architecture,
        model_name=None,
        checkpoint_path=cfg.backbone_path,
        is_train=False
    )

    loader, _ = get_dataloaders(
        data_root=cfg.data_root,
        datasets=cfg.dataset,
        train_transform=eval_transform,
        val_transform=eval_transform,
    )

    run_post_hoc_eval(cfg=cfg, backbone=backbone, dataloader=loader)

