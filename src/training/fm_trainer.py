import time
from copy import deepcopy
import dataclasses as dc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from tqdm import tqdm
import os

from src.dataloaders import ImageDataManager, get_transforms
from src.models.mae import get_mae
from src.utils.model_utils import save_model
from src.utils.visualize import visualize_batch


@dataclass
class FMTrainerConfig:
    # model args
    architecture: str = "mae-vit"
    patch_size: int = 16
    mlp_ratio: float = 4.0
    encoder_dim: int = 1024
    num_encoder_heads: int = 16
    encoder_depth: int = 24
    decoder_dim: int = 512
    num_decoder_heads: int = 16
    decoder_depth: int = 8
    norm_pix_loss: bool = True
    mask_ratio: float = 0.75
    learnable_pos_embed: bool = False

    # data args
    data_root: str = "data"
    training_datasets: str | list = field(default_factory=lambda: ["ham10000", "scin", "fitzpatrick17k"])
    dataset: str = "fp-scin-ham"
    batch_size: int = 128
    img_size: int = 224
    in_channels: int = 3
    test_size: float = 0.01
    num_workers: int = 12
    stratify: bool = True
    stratify_by: str = "label"
    transform: Any = None
    pin_memory: bool = True
    shuffle: bool = True

    # training args
    epochs: int = 400
    lr: float = 5e-4 * batch_size / 256
    optimizer: str = "adamw"
    weight_decay: float = 0.05
    beta1: float = 0.95
    beta2: float = 0.999
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 20

    do_batch_visualization: bool = False
    batch_visualization_every: int = 10
    do_online_probing: bool = True
    online_probing_every: int = 10
    probe_dataset: str = "ham10000"
    probe_dataset_2: str = "fitzpatrick17k"
    probe_classes: int = 7
    num_probe_epochs: int = 10
    probe_test_size: float = 0.2
    keep_copy_of_best_model: bool = True
    do_validation: bool = True
    # misc
    device: str = "cuda"
    seed: int = 1112
    wandb_team: str = "voerik"
    wandb_project: str = "thesis-backbone-training"
    wandb_name: str = f"{dataset}"
    output_dir: str = Path(f"./out/{dataset}-{epochs}-{encoder_dim}-{encoder_depth}-{decoder_dim}-{decoder_depth}/")

    def __repr__(self) -> str:
        """
        Provides a multi-line string representation of the config object.
        """
        fields = dc.fields(self)
        parts = [f"    {f.name}={getattr(self, f.name)!r}" for f in fields]
        body = ',\n'.join(parts)
        return f"{self.__class__.__name__}(\n{body}\n)"

    @classmethod
    def load_from_yaml(cls, config_path: str) -> 'FMTrainerConfig':
        """
        Creates a config instance from a YAML file.
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])

        if 'lr' not in config_dict and 'batch_size' in config_dict:
            config_dict['lr'] = 5e-4 * config_dict['batch_size'] / 256

        return cls(**config_dict)


class FMTrainer:
    """Trainer class for the base foundation model."""

    def __init__(self, cfg):
        self.cfg = cfg
        print("Initializing trainer...")
        self.device = self._get_device()
        if self.cfg.do_online_probing:
            datasets = [self.cfg.probe_dataset]
            datasets.extend(self.cfg.training_datasets)
            if self.cfg.probe_dataset_2:
                datasets.append(self.cfg.probe_dataset_2)
        else:
            datasets = self.cfg.training_datasets
        self.datamanager = ImageDataManager(
            data_root=self.cfg.data_root,
            initialize=datasets,
            seed=self.cfg.seed,
            transform=get_transforms(self.cfg.img_size)
        )
        self.train_loader, self.val_loader = self._get_dataloader()
        self.probe_loader_train, self.probe_loader_test = self._get_probe_loader()
        if self.cfg.probe_dataset_2:
            self.probe_loader_train_2, self.probe_loader_test_2 = self._get_probe_loader(secondary_probe=True)
        self.model = self._get_model()
        self.model.to(self.device)
        self.optimizer = self._get_optim()
        self.scheduler = self._get_scheduler()

        # ---
        self.current_best = np.inf
        self.best_epoch = 0
        self.best_model = None
        self.output_dir = self.cfg.output_dir
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_name,
            config=self.cfg,
            entity=self.cfg.wandb_team,
            settings=wandb.Settings(x_disable_stats=True)
        )        
        print(f"\tUsing device: {self.device}")
        print(f"\tUsing model: {self.cfg.architecture}")
        print(f"\tUsing optimizer: {self.cfg.optimizer}")
        print(f"\tTraining for {self.cfg.epochs} epochs")
        print(f"\tOutputs are saved to {self.output_dir}")

    def _get_device(self):
        print("\tSetting up device...")
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif self.cfg.device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def _get_scheduler(self):
        print("\tSetting up scheduler...")
        if self.cfg.scheduler == "cosine":
            from timm.scheduler import CosineLRScheduler

            return CosineLRScheduler(
                self.optimizer,
                t_initial=self.cfg.epochs,
                lr_min=self.cfg.min_lr,
                warmup_t=self.cfg.warmup_epochs,
                warmup_lr_init=1e-6,
            )
        else:
            raise NotImplementedError(
                "Scheduler not recognized. Choose from ['cosine']"
            )

    def _get_dataloader(self):
        print("\tSetting up dataset...")
        train, val = self.datamanager.get_dataloaders(
            dataset=self.cfg.training_datasets,
            test_size=self.cfg.test_size,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=self.cfg.shuffle
        )
        return train, val

    def _get_model(self):
        print("\tSetting up model...")
        return get_mae(self.cfg)

    def _get_optim(self):
        print("\tSetting up optimizer...")
        registered_optims = ["adamw"]
        if self.cfg.optimizer == "adamw":
            from timm.optim import optim_factory

            param_groups = optim_factory.param_groups_weight_decay(
                self.model, self.cfg.weight_decay
            )
            return torch.optim.AdamW(
                param_groups, lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2)
            )
        else:
            raise NotImplementedError(
                f"Optimizer not recognized. Choose from {registered_optims}"
            )

    def _calculate_duration(self, start_time, end_time):
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(hours), int(minutes), seconds

    def _run_epoch(self, mode, dataloader, epoch):
        pbar = tqdm(
            total=len(dataloader),
            bar_format="{l_bar}{bar}",
            ncols=80,
            initial=0,
            position=0,
            leave=False,
        )
        pbar.set_description(f"\t\t\t{mode}")
        self.model.mode = mode
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0

        vis_batch, vis_labels = next(iter(dataloader))

        for idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            out = self.model(imgs)

            loss, mse, ssim, psnr, lpips = (
                out["loss"],
                out["eval_metrics"]["mse"],
                out["eval_metrics"]["ssim"],
                out["eval_metrics"]["psnr"],
                out["eval_metrics"]["lpips"],
            )

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.detach().cpu()
            total_mse += mse.detach().cpu()
            total_ssim += ssim.detach().cpu()
            total_psnr += psnr.detach().cpu()
            if lpips is not None:
                total_lpips += lpips.detach().cpu()

            pbar.update(1)

        denominator = len(dataloader)
        total_loss /= denominator
        total_mse /= denominator
        total_ssim /= denominator
        total_psnr /= denominator
        if total_lpips is not None:
            total_lpips /= denominator

        print("\n")
        print(f"\t\t\t{mode.upper()} Total Loss: {total_loss}")
        print(f"\t\t\t{mode.upper()} MSE: {total_mse}")
        print(f"\t\t\t{mode.upper()} SSIM: {total_ssim}")
        print(f"\t\t\t{mode.upper()} PSNR: {total_psnr}")
        print(f"\t\t\t{mode.upper()} LPIPS: {total_lpips}")

        if self.cfg.do_batch_visualization:
            os.makedirs(f"{self.output_dir}/plots/reconstructions/", exist_ok=True)
            if epoch % self.cfg.batch_visualization_every == 0:
                visualize_batch(
                    vis_batch,
                    vis_labels["label"],
                    save_path=f"{self.output_dir}/plots/reconstructions/{epoch}-original",
                    visualize=False,
                    title=f"{epoch}-original",
                )
                out = self.model(vis_batch.to(self.device))
                recon = self.model.unpatchify(out["reconstructed_patches"])
                visualize_batch(
                    recon,
                    vis_labels["label"],
                    save_path=f"{self.output_dir}/plots/reconstructions/{epoch}-reconstruction",
                    visualize=False,
                    title=f"{epoch}-recon",
                )
        probe_acc_main, probe_acc_second  = None, None
        if self.cfg.do_online_probing and mode == "train":
            if epoch % self.cfg.online_probing_every == 0:
                probe_acc_main = self._run_online_probe(
                    dataset="probe_dataset_1",
                    label_type="label",
                    num_classes=7,
                )

                probe_acc_second = self._run_online_probe(
                    dataset="probe_dataset_2",
                    label_type="label",
                    num_classes=114
                )

        wandb.log(
            {
                f"{mode} Total Loss": total_loss,
                f"{mode} MSE": total_mse,
                f"{mode} SSIM": total_ssim,
                f"{mode} PSNR": total_psnr,
                f"{mode} LPIPS": total_lpips,
                f"{mode} Probe Acc Label": probe_acc_main,
                f"{mode} Probe Acc Secondary": probe_acc_second,
            },
            step=epoch,
        )

        if mode == "val":
            if total_loss < self.current_best:
                self.best_epoch = epoch
                self.current_best = total_loss
                if self.cfg.keep_copy_of_best_model:
                    self.best_model = deepcopy(self.model)

            print(f"\t\t\tCurrent best loss: {self.current_best}")
            print(f"\t\t\tCurrent best epoch: {self.best_epoch}")

    def run(self):
        """Runs training."""
        print(f"Start training for a maximum of {self.cfg.epochs} epochs")
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()
            print(f"\t\tEpoch {epoch+1}/{self.cfg.epochs}:")
            self._run_epoch(mode="train", dataloader=self.train_loader, epoch=epoch)

            if self.cfg.do_validation:
                with torch.no_grad():
                    self._run_epoch(mode="val", dataloader=self.val_loader, epoch=epoch)
            self.scheduler.step(epoch)
            epoch_end = time.time()
            hours, minutes, seconds = self._calculate_duration(epoch_start, epoch_end)
            print(
                f"\t\t\tTime for epoch {epoch+1}/{self.cfg.epochs}: {hours:0>2}:{minutes:0>2}:{seconds:05.2f}"
            )
        save_model(
            self.best_model,
            f"{self.cfg.architecture}-{self.cfg.dataset}.pth",
            self.output_dir / "models",
        )
        end = time.time()
        hours, minutes, seconds = self._calculate_duration(start, end)
        print(
            f"\t\t\tTraining finished after {epoch+1} epochs with a duration of {hours:0>2}:{minutes:0>2}:{seconds:05.2f}"
        )

    def _run_online_probe(self, label_type: str, num_classes: int, dataset: str = None):
        """
        Trains a linear probe for a specific label type and returns the accuracy.
        """
        print(f"\t\t Starting Online Probing for '{label_type}' ---")
        self.model.encoder.eval()


        probe_classifier = nn.Linear(self.cfg.encoder_dim, num_classes).to(self.cfg.device)
        optimizer = torch.optim.Adam(probe_classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        if dataset == "probe_dataset_1":
            probing_dataloader_train = self.probe_loader_train
            probing_dataloader_test = self.probe_loader_test
        else:
            probing_dataloader_train = self.probe_loader_train_2
            probing_dataloader_test = self.probe_loader_test_2 
        num_probe_epochs = self.cfg.num_probe_epochs

        pbar = tqdm(
            total=len(probing_dataloader_train),
            bar_format="{l_bar}{bar}",
            ncols=80,
            initial=0,
            position=0,
            leave=False,
        )
        pbar.set_description(f"\t\t\t Probing")
        for probe_epoch in range(num_probe_epochs):
            probe_classifier.train()

            for images, labels_dict in probing_dataloader_train:
                labels = labels_dict[label_type].to(self.cfg.device)
                images = images.to(self.cfg.device)

                with torch.no_grad():
                    all_patch_features = self.model.encoder(images, mask_ratio=0)
                    features = torch.mean(all_patch_features, dim=1)

                logits = probe_classifier(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)

        print(f"\n\t\t Evaluating the trained probe...")
        probe_classifier.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for images, labels_dict in probing_dataloader_test:
                labels = labels_dict[label_type].to(self.cfg.device)
                images = images.to(self.cfg.device)

                all_patch_features = self.model.encoder(images, mask_ratio=0)
                features = torch.mean(all_patch_features, dim=1)
                logits = probe_classifier(features)

                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"\t\t Probing for '{label_type}' Finished. Accuracy: {accuracy:.4f} ---")

        self.model.encoder.train()
        return accuracy
    
        
    def _get_probe_loader(self, secondary_probe: bool = False):
        print("\tSetting up probe dataset...")
        if secondary_probe:
            probe_dataset = self.cfg.probe_dataset_2
        else:
            probe_dataset = self.cfg.probe_dataset
        train, test = self.datamanager.get_dataloaders(
                dataset=probe_dataset,
                test_size=self.cfg.probe_test_size,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                shuffle=self.cfg.shuffle
                )
        return train, test
