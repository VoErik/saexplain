from sae.registry import register_sae_class, register_sae_training_class
from sae.architectures.relu import ReLUSAE, ReLUSAEConfig, ReLUTrainingSAE, ReLUSAETrainingConfig
from sae.architectures.topk import TopKSAE, TopKSAEConfig, TopKTrainingSAE, TopKSAETrainingConfig
from sae.architectures.jumprelu import JumpReLUSAE, JumpReLUSAEConfig, JumpReLUTrainingSAE, JumpReLUSAETrainingConfig
from sae.architectures.batchtopk import BatchTopKTrainingSAE, BatchTopKSAETrainingConfig
from sae.architectures.matryoshka import MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKSAETrainingConfig

register_sae_class("relu", ReLUSAE, ReLUSAEConfig)
register_sae_training_class("relu", ReLUTrainingSAE, ReLUSAETrainingConfig)

register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKSAETrainingConfig)

register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUSAETrainingConfig)

register_sae_training_class("batchtopk", BatchTopKTrainingSAE, BatchTopKSAETrainingConfig)
register_sae_training_class("matryoshka", MatryoshkaBatchTopKTrainingSAE, MatryoshkaBatchTopKSAETrainingConfig)