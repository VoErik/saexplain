import argparse

from torch import nn

from src.training.resnet_trainer import ResNetTrainerConfig, ResNetTrainer, get_fp_classweights

parser = argparse.ArgumentParser()

if __name__ == "__main__":
    for labelkey in [("label", 114)]:
        for arch in ["resnet18", "resnet50"]:
            for seed in [4, 5, 6, 7, 8]:
                cfg = ResNetTrainerConfig(
                    early_stopping_patience=10,
                    epochs=200,
                    seed=seed,
                    arch=arch,
                    label_key=labelkey[0],
                    num_classes=labelkey[1],
                    loss_fn=nn.CrossEntropyLoss(
                        label_smoothing=0.1, weight=get_fp_classweights("cuda", labelkey[0])
                    )
                )
                trainer = ResNetTrainer(cfg=cfg)
                trainer.train()