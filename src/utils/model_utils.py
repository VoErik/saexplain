import warnings

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS


with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from torchmetrics.functional import structural_similarity_index_measure

import os


def save_model(model: torch.nn.Module, model_name: str, save_dir: str = "models"):
    """
    Saves the state_dict of a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        model_name (str): The name for the saved model file (e.g., 'my_model.pth').
        save_dir (str): The directory where the model will be saved.
                         Defaults to 'models'.
    """
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, model_name)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")


def sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float()


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of images to a sequence of flattened patches."""
    return rearrange(imgs,
                     'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size,
                     p2=patch_size)

def unpatchify(reconstructed_patches: torch.Tensor, patch_size: int):
    p = patch_size
    h = w = int(reconstructed_patches.shape[1] ** 0.5)
    assert h * w == reconstructed_patches.shape[1]
    return rearrange(reconstructed_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, p1=p, p2=p)

def calculate_recon_loss(xhat, x, ids_masked, patch_size=16, mode: str = "train"):
    """
    Calculates reconstruction metrics on the masked patches only.
    Args:
        xhat: The reconstructed images (N, C, H, W).
        x: The original ground-truth images (N, C, H, W).
        ids_masked: Indices of the masked patches (N, num_masked).
        patch_size: The size of the patches.
    """
    xhat = torch.clamp(xhat, 0., 1.)
    x = torch.clamp(x, 0., 1.)

    gt_patches = patchify(x, patch_size)
    recons_patches = patchify(xhat, patch_size)

    batch_size, num_masked = ids_masked.shape[0], ids_masked.shape[1]
    ids_masked_expanded = ids_masked.unsqueeze(-1).expand(batch_size, num_masked, gt_patches.shape[-1])

    gt_masked_patches = torch.gather(gt_patches, 1, ids_masked_expanded)
    recons_masked_patches = torch.gather(recons_patches, 1, ids_masked_expanded)

    masked_mse = F.mse_loss(recons_masked_patches, gt_masked_patches)

    if masked_mse < 1e-15:
        masked_psnr = torch.tensor(100.0, device=x.device)
    else:
        # Assumes max pixel value is 1.0
        masked_psnr = 20 * torch.log10(1.0 / torch.sqrt(masked_mse))

    num_total_patches = gt_patches.shape[1]
    binary_patch_mask = torch.zeros(batch_size, num_total_patches, device=x.device)
    binary_patch_mask.scatter_(1, ids_masked, 1) # Set masked positions to 1

    # Expand the mask to create full patches
    # Shape changes from [B, N] -> [B, N, 1] -> [B, N, P*P]
    mask_patches = binary_patch_mask.unsqueeze(-1).repeat(1, 1, patch_size**2)

    mask_img = unpatchify(mask_patches, patch_size)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(x.device)
    masked_ssim = ssim_metric(xhat * mask_img, x * mask_img)
    masked_lpips = None
    if mode == "val":
        # Reshape patches to be a batch of small images (B*T, C, P, P)
        lpips_metric = LPIPS(net_type='alex', normalize=True).to(x.device)
        masked_lpips = lpips_metric(xhat * mask_img, x * mask_img)

    return {"mse": masked_mse, "ssim": masked_ssim, "psnr": masked_psnr, "lpips": masked_lpips}

def load_encoder(ckpt_path, configuration):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    encoder_state_dict = {}
    prefix = 'encoder.'
    for key, value in checkpoint.items():
        if key.startswith(prefix):
            # Remove the prefix (e.g., 'encoder.blocks.0.norm.weight' -> 'blocks.0.norm.weight')
            new_key = key.replace(prefix, '', 1)
            encoder_state_dict[new_key] = value
            
    from src.models.layers.vit_encoder import ViTEncoder
    encoder = ViTEncoder(**configuration)
    encoder.load_state_dict(encoder_state_dict, strict=False)
    return encoder