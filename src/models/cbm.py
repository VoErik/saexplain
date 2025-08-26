import torch
import torch.nn as nn
from typing import Any, Optional
import torchvision.models as models
from dataclasses import dataclass
from src.models.layers import (
    CB_Layer,
    CB_Classifier
)

def get_backbone(backbone_name: str, pretrained: bool):
    """
    Initializes a pretrained backbone and prepares it for feature extraction.
    Args:
        backbone_name (str): The name of the backbone model (e.g., 'resnet18').
        pretrained (bool): If True, returns a model pretrained on ImageNet.
    Returns:
        A tuple containing:
        - The backbone model with its final layer replaced by an Identity layer.
        - The number of output features from the backbone.
    """
    if backbone_name == 'resnet18':
        if pretrained:
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18()
        embedding_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, int(embedding_dim)
    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported.")

@dataclass
class CBMConfig:
    feature_extractor_name: str = "resnet18"
    feature_extractor_pretrained: bool = True
    cb_layer_n_concepts: int = 48
    cb_layer_depth: int = 0
    num_classes: int = 114


class CBM(nn.Module):
    """Concept Bottleneck Model."""
    feature_extractor: Any
    cb_layer: CB_Layer
    classifier: CB_Classifier

    def __init__(
            self,
            cfg,
            feature_extractor: Optional[Any] = None,
            cb_layer: Optional[CB_Layer] = None,
            classifier: Optional[CB_Classifier] = None
    ):

        super().__init__()
        self.cfg = cfg
        if feature_extractor:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor, self.embedding_dim = self.setup_feature_extractor()

        if cb_layer:
            self.cb_layer = cb_layer
        else:
            self.cb_layer = self.setup_cb_layer()

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = self.setup_classifier()

    def setup_feature_extractor(self):
        fe, embedding_dim = get_backbone(
            backbone_name=self.cfg.feature_extractor_name,
            pretrained=self.cfg.feature_extractor_pretrained
        )
        return fe, embedding_dim

    def setup_classifier(self):
        clf = CB_Classifier(
            num_classes=self.cfg.num_classes,
            num_concepts=self.cfg.cb_layer_n_concepts,
        )
        return clf

    def setup_cb_layer(self):
        cb_layer = CB_Layer(
            input_dim=self.embedding_dim,
            num_layers=self.cfg.cb_layer_depth,
            num_concepts=self.cfg.cb_layer_n_concepts,
        )
        return cb_layer

    def forward(self, x):
        embedding = self.feature_extractor(x)
        predicted_concepts = self.cb_layer(embedding)
        out = self.classifier(predicted_concepts)
        return out

    def predict_concepts(self, x):
        embedding = self.feature_extractor(x)
        return self.cb_layer(embedding)

    def predict_human_understandable_concepts(self, x):
        embedding = self.feature_extractor(x)
        return torch.nn.Sigmoid(self.cb_layer(embedding))


if __name__ == "__main__":
    cfg = CBMConfig()
    cbm = CBM(cfg)

    print(cbm)
