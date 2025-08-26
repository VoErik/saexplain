import torch.nn as nn

class CB_Classifier(nn.Linear):
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__(num_concepts, num_classes, bias=True)

    def forward(self, x):
        return super().forward(x)
