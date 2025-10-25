from src.sae.registry import register_sae_class, register_sae_training_class
from src.sae.relu import ReLUSAE, ReLUSAEConfig, ReLUTrainingSAE, ReLUSAETrainingConfig

register_sae_class("relu", ReLUSAE, ReLUSAEConfig)
register_sae_training_class("relu", ReLUTrainingSAE, ReLUSAETrainingConfig)