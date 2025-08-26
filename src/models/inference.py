from typing import Any, Optional
import torch
import torch.nn as nn
from PIL import Image

from src.models.layers.vit_encoder import ViTEncoder
from src.utils.load_backbone import load_encoder


class InferenceModel(nn.Module):
    """
    An inference model that combines a ViT encoder, an optional Sparse
    Autoencoder (SAE), and a classifier head.

    Designed for interpretability and traceability from logits back to the input.
    """
    def __init__(
            self,
            encoder: ViTEncoder | Any,
            encoder_path: str = "../dinov3",
            classifier: Optional[nn.Module] = None,
            sae: Optional[nn.Module] = None,
            device: str = "cuda",
            **kwargs
    ):
        super().__init__()
        self.device = device
        backbone, transform = load_encoder(encoder, encoder_path, device=self.device, **kwargs)
        self.encoder = backbone
        self.preprocessor = self._get_preprocessor(transform)
        self.classifier = classifier
        self.sae = sae

    def extract_encoder_features(self, image, return_patch_embeddings: bool = True):
        """
        Return pooled embeddings and optionally patch embeddings from the encoder.
        """
        image = self.preprocessor(image)
        image = image.to(self.device)

        if "clip" in str(type(self.encoder)).lower():
            self.encoder.visual.output_tokens = True
            pooled_embedding, patch_embeddings = self.encoder.visual.forward(image)
            if return_patch_embeddings:
                return pooled_embedding, patch_embeddings
            return pooled_embedding

        elif "dino" in str(type(self.encoder)).lower():
            with torch.no_grad():
                x = self.encoder.forward_features(image)

            pooled_embedding = x["x_norm_clstoken"]
            patch_embeddings = x["x_norm_patchtokens"]

            if return_patch_embeddings:
                return pooled_embedding, patch_embeddings
            return pooled_embedding

        else:
            raise ValueError(f"Encoder type {type(self.encoder)} not supported for feature extraction.")


    def _get_preprocessor(self, transform):
        def preprocessor(image_path: str):
            try:
                with Image.open(image_path) as img:
                    img_rgb = img.convert("RGB")
                    transformed_image = transform(img_rgb)
                    transformed_image = transformed_image.unsqueeze(0)
                    print(transformed_image.shape)
                    return transformed_image
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                return None
            except Exception as e:
                print(f"An error occurred while processing '{image_path}': {e}")
                return None

        return preprocessor

    def extract_sae_features(self, image: torch.Tensor):
        _, patch_embeddings = self.extract_encoder_features(image)
        sae_features = self.sae.encode(patch_embeddings)
        return sae_features

    def forward(self, image):
        """
        Performs a full forward pass from image to classification logits.
        """
        pooled_embedding, patch_embeddings = self.extract_encoder_features(image, return_patch_embeddings=True)

        if self.sae is not None:
            sae_features = self.sae.encode(patch_embeddings) # Shape: [B, num_patches, sae_dim]
            final_embedding = torch.sum(sae_features, dim=1) # Shape: [B, sae_dim] TODO: WRONG!!!!
        else:
            final_embedding = torch.mean(patch_embeddings, dim=1) # Shape: [B, enc_dim]

        if self.classifier is not None:
            out = self.classifier(final_embedding)
        else:
            out = final_embedding

        return out

