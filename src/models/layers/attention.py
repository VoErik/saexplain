from torch import nn

from src.utils.hookpoint import HookPoint


class Attention(nn.Module):
    """Multi-Head Self-Attention block."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # set up HookPoints for easier access to the models internals.
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()   # The (N, H, T, T) attention pattern
        self.hook_z = HookPoint()         # The (N, T, H*D_h) output of attention heads

    def forward(self, x):
        B, T, C = x.shape # Batch, Tokens, Channels

        qkv = self.to_qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_pattern = attn_scores.softmax(dim=-1)

        attn_pattern = self.hook_pattern(attn_pattern)

        z = attn_pattern @ v

        z = self.hook_z(z)

        z = z.transpose(1, 2).reshape(B, T, C)
        out = self.proj(z)
        return out