import torch.nn as nn

class CB_Layer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_concepts: int,
            num_layers: int = 0,
            bias: bool = True
    ):
        super().__init__()
        layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_dim,
                    out_features=num_concepts,
                    bias=bias
                ),
            ]
        )
        for _ in range(num_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(
                in_features=num_concepts,
                out_features=num_concepts,
                bias=bias)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)