from torch import nn

from src.models.layers.attention import Attention
from src.models.layers.mlp import MLP
from src.utils.hookpoint import HookPoint


class ViTBlock(nn.Module):
    """A single Transformer Encoder block for a ViT."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(dim, num_heads)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

        # set up HookPoints for easier access to the models internals
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + self.hook_attn_out(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.hook_mlp_out(mlp_out)
        x = self.hook_resid_post(x)
        return x