import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from PIL import Image
import requests
import torchvision.transforms as T
from matplotlib import pyplot as plt

from src.dataloaders.transforms import DATASET_STATISTICS
from src.models.layers.vit_decoder import ViTDecoder
from src.models.layers.vit_encoder import ViTEncoder
from src.utils.model_utils import calculate_recon_loss


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder class.

    Args:
        encoder: The encoder that is used. Must be a Vision Transformer.
        decoder: The decoder that is used. Must be a Vision Transformer.
        mask_ratio: The amount of masking that should take place. Defaults to 0.75 as suggested in the original MAE
                    paper.
        norm_pix_loss: Whether to normalize patch-wise. Defaults to True. Recent papers have shown results where not
                       normalizing at all sometimes leads to better results.
    """
    def __init__(self,
                 encoder: 'ViTEncoder',
                 decoder: 'ViTDecoder',
                 mask_ratio=0.75,
                 norm_pix_loss=True
                 ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.norm_pix_loss = norm_pix_loss
        self.mode = "train"

    def _get_target_patches(self, imgs: torch.Tensor, ids_masked: torch.Tensor):
        patch_size = self.encoder.patch_size
        target_patches = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        masked_target_pixels = torch.gather(
            target_patches, dim=1, index=repeat(ids_masked, 'b t -> b t d', d=target_patches.shape[-1])
        )
        return masked_target_pixels

    def unpatchify(self, reconstructed_patches: torch.Tensor):
        p = self.encoder.patch_size
        h = w = int(reconstructed_patches.shape[1] ** 0.5)
        assert h * w == reconstructed_patches.shape[1]
        return rearrange(reconstructed_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, p1=p, p2=p)

    def forward(self, imgs: torch.Tensor, target: torch.Tensor = None):
        """
        The main forward pass for the MAE.

        This method performs the following steps:
        1. Encodes a sparse set of visible patches from the input image.
        2. Decodes the full set of patches from the encoded representations.
        3. Calculates the training loss on the masked patches. If `norm_pix_loss`
           is True, the loss is computed on per-patch-normalized pixels.
        4. Reconstructs the full image, denormalizing it if necessary.

        Args:
            imgs: A batch of input images with pixel values in the [0, 1] range.
                  Shape: (batch_size, channels, height, width).
            target: The same as imgs but without any augmentations. The idea is to force the model to learn the actual
                  anatomical structures instead of some irrelevant artifacts.

        Returns:
            A dictionary containing the training loss, the reconstructed images,
            and other useful metadata like patch IDs.
        """

        if target is None:
            target = imgs

        encoded_patches, ids_restore, ids_masked = self.encoder(
            imgs,
            mask_ratio=self.mask_ratio,
            return_ids=True
        )
        reconstructed_patches_pixels = self.decoder(encoded_patches, ids_restore)

        num_patches = reconstructed_patches_pixels.shape[1]
        all_patch_ids = torch.arange(num_patches).to(imgs.device).unsqueeze(0).expand(imgs.shape[0], -1)

        if self.norm_pix_loss:
            all_target_patches = self._get_target_patches(target, all_patch_ids)
            mean = all_target_patches.mean(dim=-1, keepdim=True)
            var = all_target_patches.var(dim=-1, keepdim=True)

            target_all_normalized = (all_target_patches - mean) / ((var + 1e-6) ** 0.5)

            target_for_loss = torch.gather(
                target_all_normalized, dim=1,
                index=repeat(ids_masked, 'b t -> b t d', d=target_all_normalized.shape[-1])
            )
        else:
            target_for_loss = self._get_target_patches(target, ids_masked)

        predicted_masked_pixels = torch.gather(
            reconstructed_patches_pixels,
            dim=1,
            index=repeat(ids_masked, 'b t -> b t d', d=reconstructed_patches_pixels.shape[-1])
        )

        loss = F.mse_loss(predicted_masked_pixels, target_for_loss)

        if self.norm_pix_loss:
            reconstructed_denorm = reconstructed_patches_pixels * ((var + 1e-6) ** 0.5) + mean
            reconstructed_imgs = self.unpatchify(reconstructed_denorm)
            return_patches = reconstructed_denorm
        else:
            reconstructed_imgs = self.unpatchify(reconstructed_patches_pixels)
            return_patches = reconstructed_patches_pixels

        reconstructed_imgs = torch.clamp(reconstructed_imgs, 0., 1.)

        with torch.no_grad():
            eval_metrics = calculate_recon_loss(
                xhat=reconstructed_imgs,
                x=target,
                ids_masked=ids_masked,
                patch_size=self.encoder.patch_size,
                mode=self.mode,
            )

        output_dict = {
            "loss": loss,
            "reconstructed_imgs": reconstructed_imgs,
            "reconstructed_patches": return_patches,
            "ids_restore": ids_restore,
            "ids_masked": ids_masked,
            "eval_metrics": eval_metrics
        }

        return output_dict


    def visualize_reconstruction(
            self, original_img, reconstructed_patches, ids_masked, title_prefix="Plot", save: bool = False
    ):
        """
        Visualizes reconstruction. Assumes input `original_img` is in [0, 1] range.
        The reconstructed image is clamped to [0, 1] for visualization.
        """
        img_for_display = torch.clamp(original_img.squeeze(0).cpu(), 0, 1).permute(1, 2, 0).numpy()

        masked_img_display = original_img.squeeze(0).clone().cpu()
        patches = rearrange(
            masked_img_display, 'c (h p1) (w p2) -> (h w) c p1 p2',
            p1=self.encoder.patch_size, p2=self.encoder.patch_size
        )
        patches[ids_masked.squeeze(0).cpu()] = 0 # Black out masked patches
        masked_img_display = rearrange(patches, '(h w) c p1 p2 -> c (h p1) (w p2)', h=int(patches.shape[0]**0.5))
        masked_img_display = masked_img_display.permute(1, 2, 0).numpy()

        reconstructed_img = self.unpatchify(reconstructed_patches.cpu()).squeeze(0).detach()
        reconstructed_img = torch.clamp(reconstructed_img, 0, 1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{title_prefix} Mask Ratio: {self.mask_ratio}", fontsize=16)
        plt.subplot(1, 3, 1); plt.imshow(img_for_display); plt.title("Original"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(masked_img_display); plt.title("Masked"); plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(reconstructed_img); plt.title("Reconstructed"); plt.axis('off')
        if save:
            save_path = f"plots/{title_prefix}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

def get_mae(cfg):
    enc = ViTEncoder(
        image_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        dim=cfg.encoder_dim,
        depth=cfg.encoder_depth,
        heads=cfg.num_encoder_heads,
        mlp_ratio=cfg.mlp_ratio,
        learned_pos_embed=cfg.learnable_pos_embed
    )

    dec = ViTDecoder(
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        num_patches=(cfg.img_size//cfg.patch_size)**2,
        encoder_dim=cfg.encoder_dim,
        dim=cfg.decoder_dim,
        depth=cfg.decoder_depth,
        heads=cfg.num_decoder_heads,
        mlp_ratio=cfg.mlp_ratio,
        learnable_pos_embed=cfg.learnable_pos_embed
    )
    mae = MaskedAutoencoder(
        encoder=enc,
        decoder=dec,
        mask_ratio=cfg.mask_ratio,
        norm_pix_loss=cfg.norm_pix_loss
    )
    return mae

if __name__ == '__main__':
    IMG_SIZE = 224
    PATCH_SIZE = 16
    encoder_cfg = dict(
        image_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        dim=768,
        depth=6,
        heads=8,
    )
    decoder_cfg = dict(
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_patches=(IMG_SIZE//PATCH_SIZE)**2,
        encoder_dim=768,
        dim=512,
        depth=4,
        heads=8
    )

    encoder = ViTEncoder(**encoder_cfg)
    decoder = ViTDecoder(**decoder_cfg)
    mae = MaskedAutoencoder(encoder, decoder, mask_ratio=0.75)
    print("--- MAE Model Instantiated Successfully ---")

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])
        processed_image = transform(image).unsqueeze(0)

        print(f"\nSuccessfully loaded and processed real image.")
        print(f"Image tensor shape: {processed_image.shape}")

        mae.eval()
        with torch.no_grad():
            out = mae(processed_image)

        loss = out["loss"]
        eval_metrics = out["eval_metrics"]
        reconstructed_patches = out["reconstructed_patches"]
        ids_masked = out["ids_masked"]

        print(f"\n--- Forward Pass Test ---")
        print(f"Successfully completed forward pass with real image.")
        print(f"Calculated Loss: {loss.item():.4f}")
        print(f"Eval Metrics: {eval_metrics}")

        print("\n--- Shape Verification ---")
        reconstructed_images = mae.unpatchify(reconstructed_patches)
        print(f"Shape of reconstructed images: {reconstructed_images.shape}")
        assert reconstructed_images.shape == processed_image.shape, \
            "Shape mismatch between original and reconstructed image!"
        print("SUCCESS: Reconstructed image shape matches original image shape.")

        print("\n--- Visualization ---")
        print("Displaying original, masked, and reconstructed images...")
        mae.visualize_reconstruction(
            processed_image, reconstructed_patches, ids_masked, title_prefix="Untrained Model"
        )

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
