import argparse
from pathlib import Path

import torch
import gc

from src.models.sae.activation_store import VisionActivationStore
from src.models.sae.config import TrainingSAEConfig
from src.models.sae.trainer import SAETrainer, get_training_sae
from src.models.sae.utils import load_configs_from_yaml, get_data_center

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/sae.yaml')
parser.add_argument('--device', default='cuda')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./checkpoints/fitzpatrick17k/')

if __name__ == '__main__':
    args = parser.parse_args()
    cfg_dicts = load_configs_from_yaml(args.config)
    device = torch.device(args.device)
    print("Training on ", device)
    ARCHS = ["gated", "jumprelu", "topk", "batchtopk", "standard"]
    EXPANSION_FACTORS = [16]
    k = 64

    for arch in ARCHS:
        for expansion in EXPANSION_FACTORS:
            sae_cfg = TrainingSAEConfig.from_dict(cfg_dicts["sae"])

            sae_cfg.architecture = arch
            if arch in ["standard", "gated"]:
                sae_cfg.activation_fn_str = "relu"
                sae_cfg.activation_fn_kwargs = {}
            elif arch == "jumprelu":
                sae_cfg.activation_fn_str = "jumprelu"
                sae_cfg.activation_fn_kwargs["alpha"] = 0.01
            elif arch in ["topk", "batchtopk"]:
                sae_cfg.activation_fn_kwargs["k"] = k
            sae_cfg.d_sae = sae_cfg.d_in * expansion


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

            if sae_cfg.b_dec_init_method != "zeros":
                data_center = get_data_center(train_activation_store, sae_cfg.b_dec_init_method)
                sae_model.b_dec.data = data_center.to(device)
                print(f"Initialized b_dec with {sae_cfg.b_dec_init_method}.")

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

            save_dir = Path(sae_cfg.checkpoint_path)
            sae_model.save_model(save_dir / sae_model.get_name())
            print(f"Model saved to {save_dir}.")

            print("Clearing memory for next run...")
            del sae_model
            del trainer
            del train_activation_store
            del eval_activation_store

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

