import torch
import torch.nn as nn
from einops import repeat

from src.models.layers.block import ViTBlock
from src.models.layers.patch_embedding import PatchEmbedding, PositionalEmbedding
from src.utils.hooked_rootmodule import HookedRootModule
from src.utils.model_utils import sincos_pos_embed

class ViTEncoder(HookedRootModule):
    """
    A self-contained, hookable Vision Transformer Encoder.
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 dim=768,
                 depth=12,
                 heads=12,
                 mlp_ratio=4.0,
                 learned_pos_embed: bool = False
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        if learned_pos_embed:
            self.pos_embed = PositionalEmbedding(self.patch_embed.num_patches, dim)
        else:
            gridsize = image_size // patch_size
            self.pos_embed = sincos_pos_embed(embed_dim=dim, grid_size=gridsize) # img_size // patch_size
        self.blocks = nn.ModuleList([
            ViTBlock(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.setup()

    # TODO: I don't think it's good to do this here. Refactor to let MAE class handle masking.
    def _random_masking(self, patches: torch.Tensor, mask_ratio: float):
        """
        Helper function to perform masking.
        """
        B, T, D = patches.shape
        device = patches.device
        num_visible = int(T * (1 - mask_ratio))

        noise = torch.rand(B, T, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_visible = ids_shuffle[:, :num_visible]
        ids_masked = ids_shuffle[:, num_visible:]

        visible_patches = torch.gather(patches, dim=1, index=repeat(ids_visible, 'b t -> b t d', d=D))

        return visible_patches, ids_visible, ids_masked, ids_restore

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75, return_ids: bool = False):
        """
        The forward method.

        Args:
            imgs (torch.Tensor): Input images (B, C, H, W).
            mask_ratio (float): The fraction of patches to mask.
            return_ids (bool): If True, also returns the restoration indices for a decoder.
        """
        patch_embeds = self.patch_embed(imgs)

        visible_patches, ids_visible, ids_masked, ids_restore = self._random_masking(patch_embeds, mask_ratio)

        if isinstance(self.pos_embed, nn.Module):
            full_pos_embed = self.pos_embed()
            pos_embed_expanded = full_pos_embed.expand(visible_patches.shape[0], -1, -1)
        else:
            full_pos_embed = self.pos_embed.to(visible_patches.device)
            pos_embed_expanded = full_pos_embed.unsqueeze(0).expand(visible_patches.shape[0], -1, -1)


        pos_embed_visible = pos_embed_expanded.gather(
            1,
            repeat(ids_visible, 'b v -> b v d', d=visible_patches.shape[-1])
        )

        x = visible_patches + pos_embed_visible

        for block in self.blocks:
            x = block(x)

        encoder_output = self.norm(x)

        if return_ids:
            return encoder_output, ids_restore, ids_masked
        return encoder_output
