import wandb
from sae.core import TrainingSAE, TrainingSAEConfig
from sae.trainer import TrainingRunnerConfig, SAETrainerConfig, SAETrainer, LoggingConfig
from sae.eval import SAEEvaluator, EvaluationConfig


def train_sae(cfg_dict: dict):
    cfg = TrainingRunnerConfig.from_dict(cfg_dict)
    print("Running training with:")
    print(cfg)
    
    if cfg.log_to_wandb:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=cfg.to_dict(),
                name=cfg.run_name,
                id=cfg.wandb_id,
            )
    sae = TrainingSAE.from_dict(
        TrainingSAEConfig.from_dict(
            cfg.to_dict()
        ).to_dict()
    )

    from src.utils.embedding_cache_with_labels import EmbeddingCache
    cache = EmbeddingCache.from_dict(
        cfg.to_dict()
    )
   
    evaluator = SAEEvaluator(
         EvaluationConfig.from_dict(
              cfg.to_dict()
            )
        )

    trainer_cfg = SAETrainerConfig.from_dict(
        cfg.to_dict()
    )
    trainer_cfg.logger = LoggingConfig.from_dict(cfg_dict)
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=cache.train_dataset,
        eval_dataset=cache.eval_dataset,
        evaluator=evaluator
    )

    _ = trainer.fit()

    if cfg.log_to_wandb:
        wandb.finish()