import streamlit as st
import numpy as np
import json
import torch
import umap
from PIL import Image, ImageDraw, ImageEnhance
from typing import Dict, List, Any

from matplotlib import cm

from src.models.sae.sae_utils import aggregate_activations


@st.cache_data
def load_data(
        atlas_path: str,
        clusters_path: str
):
    """Loads the atlas and cluster data from their respective files."""
    try:
        npz_file = np.load(atlas_path, allow_pickle=True)
        atlas = {int(k): v.tolist() for k, v in npz_file.items()}
    except FileNotFoundError:
        st.error(f"Atlas file not found at: {atlas_path}")
        return None, None

    try:
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
    except FileNotFoundError:
        st.error(f"Clusters file not found at: {clusters_path}")
        return atlas, None

    return atlas, clusters

def save_atlas_npz(
        atlas_path: str,
        atlas_data: Dict
):
    """Saves the atlas dictionary to a compressed .npz file."""
    np.savez_compressed(
        atlas_path,
        **{str(k): np.array(v, dtype=object) for k, v in atlas_data.items()}
    )

def create_dimmed_mask_image(
        img_path: str,
        active_indices: List[int],
        feat_extractor_info: Dict,
        dim_factor: float = 0.7
):
    """Masks out non-active patches by dimming them."""
    try:
        original_image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        return None
    img_width, img_height = original_image.size
    grid_h, grid_w = feat_extractor_info['grid_size']

    if grid_h == 0 or grid_w == 0:
        return original_image

    patch_size_w = img_width // grid_w
    patch_size_h = img_height // grid_h
    mask = Image.new('L', (img_width, img_height), color=0)
    mask_draw = ImageDraw.Draw(mask)

    for patch_index in active_indices:
        row, col = patch_index // grid_w, patch_index % grid_w
        x1, y1 = col * patch_size_w, row * patch_size_h
        x2, y2 = x1 + patch_size_w, y1 + patch_size_h
        mask_draw.rectangle([x1, y1, x2, y2], fill=255)

    enhancer = ImageEnhance.Brightness(original_image)
    dimmed_image = enhancer.enhance(1.0 - dim_factor)
    return Image.composite(original_image, dimmed_image, mask)

def create_binary_mask_image(
        img_path: str,
        active_indices: List[int],
        feat_extractor_info: Dict
):
    """Shows only the active patches on a black background."""
    try:
        original_image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        return None

    background = Image.new('RGB', original_image.size, (0, 0, 0))
    img_width, img_height = original_image.size
    grid_h, grid_w = feat_extractor_info['grid_size']

    if grid_h == 0 or grid_w == 0:
        return original_image

    patch_size_w = img_width // grid_w
    patch_size_h = img_height // grid_h

    for patch_index in active_indices:
        row, col = patch_index // grid_w, patch_index % grid_w
        x1, y1 = col * patch_size_w, row * patch_size_h
        x2, y2 = x1 + patch_size_w, y1 + patch_size_h
        patch = original_image.crop((x1, y1, x2, y2))
        background.paste(patch, (x1, y1))

    return background

def create_heatmap_image(
        img_path: str,
        active_patches: List,
        feat_extractor_info: Dict
):
    """Overlays a heatmap on the image based on patch activation strength."""
    try:
        original_image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        return None

    heatmap_overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap_overlay)
    max_activation = max([val for _, val in active_patches], default=1.0)

    if max_activation == 0:
        max_activation = 1.0

    grid_h, grid_w = feat_extractor_info['grid_size']
    patch_size_h = original_image.height // grid_h
    patch_size_w = original_image.width // grid_w
    colormap = cm.get_cmap('viridis')

    for patch_index, activation_value in active_patches:
        normalized_activation = activation_value / max_activation
        color = colormap(normalized_activation)
        fill_color = tuple(int(c * 255) for c in color)[:3] + (150,)
        row, col = patch_index // grid_w, patch_index % grid_w
        x1, y1 = col * patch_size_w, row * patch_size_h
        x2, y2 = x1 + patch_size_w, y1 + patch_size_h
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

    return Image.alpha_composite(original_image.convert('RGBA'), heatmap_overlay).convert('RGB')

@st.cache_data
def load_umap_embeddings(umap_path: str):
    """Loads pre-computed UMAP embeddings from a .npy file."""
    try:
        return np.load(umap_path)
    except FileNotFoundError:
        return None

@st.cache_data
def calculate_activation_frequencies(activations_tensor_path: str):
    """Loads the full activation tensor and calculates feature frequencies."""
    try:
        all_activations = torch.load(activations_tensor_path)
        fire_counts = (all_activations > 0).sum(dim=0).float()
        return ((fire_counts / all_activations.shape[0]) * 100).cpu().numpy()
    except FileNotFoundError:
        return None


def run_inference_on_image(
        image: Image.Image,
        feature_extractor: Any,
        sae_model: Any,
        prep: Any,
        device: str,
        k: int = 10 # Todo: make this adjustable in App
):
    """
    Runs a single image through the models and returns the top-k activating features.
    """
    feature_extractor.to(device).eval()
    sae_model.to(device).eval()

    image_tensor = prep(image).unsqueeze(0).to(device) # add batch dim

    with torch.no_grad():
        if "clip" in str(type(feature_extractor)).lower():
            feature_extractor.visual.output_tokens = True
            _ , patch_embeddings = feature_extractor.visual.forward(image_tensor)
        elif "dino" in str(type(feature_extractor)).lower():
            x = feature_extractor.forward_features(image_tensor)
            patch_embeddings = x["x_norm_patchtokens"]
        else:
            patch_embeddings = feature_extractor(image_tensor)

        activations = sae_model.encode(patch_embeddings)

    item_activations = activations[0]
    image_level_scores = aggregate_activations(item_activations, 'binary_sum', 0.2)
    top_scores, top_indices = torch.topk(image_level_scores, k)

    return top_indices.cpu().tolist(), top_scores.cpu().tolist()


@st.cache_data
def calculate_mean_activations(activations_tensor_path: str):
    """
    Calculates the mean activation value for each feature, considering only non-zero activations.
    """
    try:
        all_activations = torch.load(activations_tensor_path, map_location='cpu')
        activations_with_nan = torch.where(all_activations > 0, all_activations, torch.nan)
        mean_values = torch.nanmean(activations_with_nan, dim=0)
        mean_values = torch.nan_to_num(mean_values, nan=0.0)
        return mean_values.numpy()
    except FileNotFoundError:
        return None