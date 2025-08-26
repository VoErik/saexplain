import torch
import torch.nn as nn

from src.utils.hookpoint import HookPoint

class MLP(nn.Module):
    """A standard MLP block with a GELU activation."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.hook_mlp = HookPoint()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.hook_mlp(x)
        x = self.fc2(x)
        return x
