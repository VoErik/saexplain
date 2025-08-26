import torch
import torch.nn as nn
from src.models.layers.block import ViTBlock
from src.models.layers.patch_embedding import PositionalEmbedding
from src.utils.model_utils import sincos_pos_embed


class ViTDecoder(nn.Module):
    """
    A lightweight Vision Transformer Decoder for a Masked Autoencoder.

    This module takes the latent representations of visible patches and
    reconstructs the pixel values for the *entire* image.
    """
    def __init__(self,
                 patch_size=16,
                 in_channels=3,
                 num_patches=196,
                 encoder_dim=768,
                 dim=512,
                 depth=8,
                 heads=16,
                 mlp_ratio=4.0,
                 learnable_pos_embed: bool = False):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.decoder_dim = dim

        self.embed = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.randn(dim))

        if learnable_pos_embed:
            self.pos_embed = PositionalEmbedding(num_patches, dim)
        else:
            gridsize = 224 // patch_size # TODO: hardcoded image size -> make parameter
            self.pos_embed = sincos_pos_embed(embed_dim=dim, grid_size=gridsize) # img_size // patch_size

        self.blocks = nn.ModuleList([
            ViTBlock(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)

        num_pixels_per_patch = in_channels * patch_size ** 2
        self.to_pixels = nn.Linear(dim, num_pixels_per_patch)


    def forward(self, encoded_patches: torch.Tensor, ids_restore: torch.Tensor):
        """
        Reconstructs the image from encoded patches and restoration indices.

        Args:
            encoded_patches (torch.Tensor): Latent representations of visible patches
                                            (B, num_visible, encoder_dim).
            ids_restore (torch.Tensor): The indices needed to un-shuffle the patches
                                        back to their original order (B, num_patches).
        """
        B, num_visible, _ = encoded_patches.shape
        device = encoded_patches.device

        x = self.embed(encoded_patches)

        num_masked = self.num_patches - num_visible
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        x_full_shuffled = torch.cat([x, mask_tokens], dim=1)

        x_full = torch.gather(x_full_shuffled, 1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim))

        if isinstance(self.pos_embed, nn.Module):
            pos_embed_to_add = self.pos_embed()
        else:
            pos_embed_to_add = self.pos_embed.to(device)

        x_full = x_full + pos_embed_to_add

        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        return self.to_pixels(x_full)
