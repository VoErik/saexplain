from typing import Optional

import torch
import torch.nn as nn
from typing import Optional, List

class MLP(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            num_classes: int, 
            hidden_sizes: Optional[List[int]] = None, 
            dropout_rate: float = 0.0, 
            use_input_norm: bool = False
            ):
        """
        A dynamic Multi-Layer Perceptron (MLP) classifier.

        Args:
            input_dim (int): Size of the input feature vector.
            num_classes (int): Number of output classes.
            hidden_sizes (List[int], optional): A list of integers representing the size of each hidden layer. 
                                               Defaults to None (no hidden layers, just a linear classifier).
            dropout_rate (float, optional): Probability of dropout between layers. Defaults to 0.0.
        """
        super().__init__()
        
        hidden_sizes = hidden_sizes or []
        self.norm = nn.LayerNorm(input_dim) if use_input_norm else nn.Identity()
        self.layers = self._build_layers(input_dim, num_classes, hidden_sizes, dropout_rate)

    def _build_layers(self, input_dim: int, num_classes: int, hidden_sizes: List[int], dropout_rate: float) -> nn.Sequential:
        """
        Helper method to construct the network layers dynamically.
        """
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        """
        return self.layers(x)