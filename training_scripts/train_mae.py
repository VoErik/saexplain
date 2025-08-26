from src.training.fm_trainer import FMTrainer, FMTrainerConfig
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/mae_2.yaml')

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = FMTrainerConfig.load_from_yaml(args.config)
    trainer = FMTrainer(cfg=cfg)
    trainer.run()
