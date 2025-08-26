import argparse

from src.training.cbm_trainer import CBMTrainerConfig, CBMTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sequential", choices=["sequential", "joint"])
parser.add_argument("--concept_epochs", type=int, default=100)
parser.add_argument("--classifier_epochs", type=int, default=100)
parser.add_argument("--freeze_feature_extractor", type=bool, default=False)
parser.add_argument(
    "--dataset", type=str, default="skincon_fitzpatrick17k", choices=["skincon_fitzpatrick17k", "ddi"]
)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = CBMTrainerConfig()
    trainer = CBMTrainer(cfg=cfg)
    trainer.train(method=args.method)