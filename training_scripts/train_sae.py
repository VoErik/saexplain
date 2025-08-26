import argparse
from pathlib import Path

import torch
import gc

from src.models.sae.activation_store import VisionActivationStore
from src.models.sae.config import TrainingSAEConfig
from src.models.sae.core import SAE
from src.models.sae.trainer import SAETrainer, get_training_sae
from src.models.sae.utils import load_configs_from_yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/sae.yaml')
parser.add_argument('--device', default='cuda')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./checkpoints/fitzpatrick17k/')

if __name__ == '__main__':
    args = parser.parse_args()
    cfg_dicts = load_configs_from_yaml(args.config)
    device = torch.device(args.device)

    EXPANSION_FACTORS = [32]
    KS = [8, 16, 32, 64, 128]

    for expansion in EXPANSION_FACTORS:
        for k in KS:
            sae_cfg = TrainingSAEConfig.from_dict(cfg_dicts["sae"])

            sae_cfg.d_sae = sae_cfg.d_in * expansion
            sae_cfg.activation_fn_kwargs["k"] = k

            sae_model = get_training_sae(architecture=sae_cfg.architecture, cfg=sae_cfg)
            sae_model.to(device)
            print(f"SAE Model: {sae_model.get_name()}")
            print(f"Activation function: {sae_model.activation_fn}")

            act_store_cfg = cfg_dicts["activation_store"]

            train_activation_store = VisionActivationStore(
                dataset_name=act_store_cfg["dataset_name"],
                feature_extractor_model=act_store_cfg["feature_extractor"],
                d_in=sae_cfg.d_in,
                store_batch_size=sae_cfg.train_batch_size,
                device=device,
                cache_path=Path(f"{act_store_cfg['embedding_cache_path']}"
                                f"/{act_store_cfg['dataset_name']}"
                                f"/{act_store_cfg['feature_extractor']}"
                                f"/train_embeds_d{sae_cfg.d_in}.pt"),
                extraction_batch_size=act_store_cfg["extraction_batch_size"],
                shuffle_each_epoch=True,
            )

            eval_activation_store = VisionActivationStore(
                dataset_name=act_store_cfg["eval_dataset_name"],
                feature_extractor_model=act_store_cfg["feature_extractor"],
                d_in=sae_cfg.d_in,
                device=device,
                cache_path=Path(f"{act_store_cfg['embedding_cache_path']}"
                                f"/{act_store_cfg['eval_dataset_name']}"
                                f"/{act_store_cfg['feature_extractor']}"
                                f"/eval_embeds_d{sae_cfg.d_in}.pt"),
                extraction_batch_size=act_store_cfg["extraction_batch_size"],
                shuffle_each_epoch=False,
            )

            trainer = SAETrainer(
                sae_model=sae_model,
                activation_store=train_activation_store,
                eval_activation_store=eval_activation_store
            )
            print("Starting SAE training...")
            trainer.train(num_epochs=args.num_epochs)

            save_dir = Path(args.save_dir)
            sae_model.save_model(save_dir)
            print(f"Model saved to {save_dir}.")

            loaded_sae = SAE.load_model(save_dir)

            print("Clearing memory for next run...")
            del sae_model
            del trainer
            del train_activation_store
            del eval_activation_store

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

