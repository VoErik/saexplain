from src.sae.registry import register_sae_class, register_sae_training_class
from src.sae.relu import ReLUSAE, ReLUSAEConfig, ReLUTrainingSAE, ReLUSAETrainingConfig
from src.sae.topk import TopKSAE, TopKSAEConfig, TopKTrainingSAE, TopKSAETrainingConfig
from src.sae.jumprelu import JumpReLUSAE, JumpReLUSAEConfig, JumpReLUTrainingSAE, JumpReLUSAETrainingConfig
from src.sae.batchtopk import BatchTopKTrainingSAE, BatchTopKSAETrainingConfig
from src.sae.matryoshka import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKSAETrainingConfig

register_sae_class("relu", ReLUSAE, ReLUSAEConfig)
register_sae_training_class("relu", ReLUTrainingSAE, ReLUSAETrainingConfig)

register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKSAETrainingConfig)

register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUSAETrainingConfig)

register_sae_training_class("batchtopk", BatchTopKTrainingSAE, BatchTopKSAETrainingConfig)
register_sae_training_class("matryoshka", MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKSAETrainingConfig)