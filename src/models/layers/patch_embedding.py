import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Takes a 2D image and converts it into a 1D sequence of patch embeddings.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, d_model)
        """
        # Project the image into patches: (B, C, H, W) -> (B, d_model, grid_size, grid_size)
        x = self.proj(x)

        # Flatten the spatial dimensions and transpose to get the sequence format
        # (B, d_model, grid_size, grid_size) -> (B, num_patches, d_model)
        x = rearrange(x, 'b d h w -> b (h w) d')

        return x


class PositionalEmbedding(nn.Module):
    """
    A learnable positional embedding module.
    """
    def __init__(self, num_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

    def forward(self) -> torch.Tensor:
        return self.pos_embed
