# utils.py
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.cm as cm
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import torch
import pandas as pd

@st.cache_data
def load_data(atlas_path: str, clusters_path: str):
    """
    Loads the atlas and cluster data from their respective JSON files.
    """
    atlas = None
    clusters = None
    
    try:
        with open(atlas_path, 'r') as f:
            atlas_data = json.load(f)
            atlas = {int(k.split("_")[-1]): v for k, v in atlas_data.items()}
            
    except FileNotFoundError:
        st.error(f"Atlas file not found at: {atlas_path}")
    except json.JSONDecodeError:
        st.error(f"Failed to decode atlas.json. Is the file valid?")
        
    try:
        with open(clusters_path, 'r') as f:
            clusters = json.load(f)
    except FileNotFoundError:
        st.warning(f"Clusters file not found at: {clusters_path}")
    except json.JSONDecodeError:
        st.error(f"Failed to decode sae_clusters.json. Is the file valid?")

    if atlas is None:
        return None, None
        
    return atlas, clusters

def get_sae_dirs(base_dir: str):
    """Finds all valid SAE directories within the base analysis directory."""
    sae_dirs = []
    if not os.path.isdir(base_dir):
        return []
    for dir_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dir_name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "atlas.json")):
            sae_dirs.append(dir_name)
    return sae_dirs

def create_dimmed_mask_image(
        img_path: str,
        active_indices: List[int],
        feat_extractor_info: Dict,
        dim_factor: float = 0.7,
        img_size: tuple[int, int] = (224, 224)
):
    """Masks out non-active patches by dimming them."""
    try:
        original_image = Image.open(img_path).convert("RGB")
        original_image = original_image.resize(size=img_size)
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
        feat_extractor_info: Dict,
        img_size: tuple[int, int] = (224, 224)
):
    """Shows only the active patches on a black background."""
    try:
        original_image = Image.open(img_path).convert("RGB")
        original_image = original_image.resize(size=img_size)
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
        active_patches: List[Tuple[int, float]], # List of (index, value) tuples
        feat_extractor_info: Dict,
        img_size: tuple[int, int] = (224, 224)
):
    """Overlays a heatmap on the image based on patch activation strength."""
    try:
        original_image = Image.open(img_path).convert("RGB")
        original_image = original_image.resize(size=img_size)
    except FileNotFoundError:
        return None

    heatmap_overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap_overlay)
    
    if not active_patches:
        return original_image
        
    max_activation = max([val for _, val in active_patches], default=1.0)
    if max_activation == 0:
        max_activation = 1.0

    grid_h, grid_w = feat_extractor_info['grid_size']
    patch_size_h = original_image.height // grid_h
    patch_size_w = original_image.width // grid_w
    colormap = cm.get_cmap('coolwarm')

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
def load_activation_stats(stats_dir_path: str):
    """
    Loads the pre-computed 'sparsity.pt' and 'mean_acts.pt' files
    from the SAE's analysis directory.
    """
    sparsity_path = os.path.join(stats_dir_path, "sparsity.pt")
    mean_acts_path = os.path.join(stats_dir_path, "mean_acts.pt")

    try:
        frequencies = torch.load(sparsity_path, map_location='cpu')
        mean_activations = torch.load(mean_acts_path, map_location='cpu')

    except FileNotFoundError as e:
        st.error(f"Missing required stats file: {e.filename}")
        st.info(f"Ensure 'sparsity.pt' and 'mean_acts.pt' are in: {stats_dir_path}")
        return None
    except Exception as e:
        st.error(f"Error loading stats files: {e}")
        return None

    df = pd.DataFrame({
        'feature_id': np.arange(len(frequencies)),
        'frequency': frequencies.numpy(),
        'mean_activation': mean_activations.numpy()
    })
    
    return df

@st.cache_data
def load_umap_data(stats_dir_path: str) -> Optional[pd.DataFrame]:
    """
    Loads pre-computed UMAP embeddings and merges them with activation stats.
    """
    stats_df = load_activation_stats(stats_dir_path)
    if stats_df is None:
        st.error("Could not load activation stats, UMAP plot will be incomplete.")
        return None

    umap_path = os.path.join(stats_dir_path, "umap_embeddings.pt")
    try:
        embeddings_2d = torch.load(umap_path, map_location='cpu').numpy()
    except FileNotFoundError:
        st.error(f"UMAP file not found: {umap_path}")
        st.info("Please run the 'generate_and_save_umap' function in your SAE project.")
        return None
    except Exception as e:
        st.error(f"Error loading UMAP embeddings: {e}")
        return None

    if len(embeddings_2d) != len(stats_df):
        st.error(f"Data mismatch: UMAP file has {len(embeddings_2d)} features, but stats have {len(stats_df)}.")
        return None
        
    umap_df = pd.DataFrame({
        'feature_id': np.arange(len(embeddings_2d)),
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    merged_df = pd.merge(stats_df, umap_df, on='feature_id')
    
    return merged_df