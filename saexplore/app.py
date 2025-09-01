import streamlit as st
import numpy as np
import json
from PIL import Image, ImageOps, ImageEnhance
from typing import Dict, List, Any

from PIL import ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="../analysis")
parser.add_argument("--data", type=str, default="../")


def save_atlas_npz(atlas_path: str, atlas_data: Dict):
    """Saves the atlas dictionary to a compressed .npz file."""
    np.savez_compressed(
        atlas_path,
        **{str(k): np.array(v, dtype=object) for k, v in atlas_data.items()}
    )

def create_highlighted_image(img_path: str, active_indices: List[int], feat_extractor_info: Dict, dim_factor: float = 0.7):
    """
    Masks out non-active patches by dimming them, making active patches stand out.

    Args:
        img_path (str): Path to the original image.
        active_indices (List[int]): List of indices for patches that are active.
        feat_extractor_info (Dict): Dictionary containing 'grid_size' tuple (grid_h, grid_w).
        dim_factor (float): How much to dim the inactive areas (0.0 = fully black, 1.0 = no dimming).

    Returns:
        PIL.Image.Image: The image with inactive regions dimmed, or None if image not found.
    """
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
        row = patch_index // grid_w
        col = patch_index % grid_w
        x1, y1 = col * patch_size_w, row * patch_size_h
        x2, y2 = x1 + patch_size_w, y1 + patch_size_h

        mask_draw.rectangle([x1, y1, x2, y2], fill=255) # 255 = white
    enhancer = ImageEnhance.Brightness(original_image)
    dimmed_image = enhancer.enhance(1.0 - dim_factor) # Dim by 1 - dim_factor

    final_image = Image.composite(original_image, dimmed_image, mask)

    return final_image

@st.cache_data
def load_data(atlas_path, clusters_path):
    """Loads the atlas and cluster data, converting atlas arrays to lists."""
    npz_file = np.load(atlas_path, allow_pickle=True)
    atlas = {int(k): v.tolist() for k, v in npz_file.items()}
    with open(clusters_path, 'r') as f:
        clusters = json.load(f)
    return atlas, clusters


def main(dir, datadir):
    st.set_page_config(layout="wide")
    st.title("🔬 SAE Atlas Explorer")

    ATLAS_PATH = f"{dir}/sae_atlas.npz"
    CLUSTERS_PATH = f"{dir}/sae_clusters.json"

    if 'atlas' not in st.session_state:
        try:
            atlas_data, cluster_data = load_data(ATLAS_PATH, CLUSTERS_PATH)
            st.session_state.atlas = atlas_data
            st.session_state.clusters = cluster_data
        except FileNotFoundError as e:
            st.error(f"Error loading data: {e}. Make sure files are in the correct paths.")
            st.stop()

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Explorer", ["Single Feature Explorer", "Cluster Explorer"])


    if app_mode == "Single Feature Explorer":
        st.sidebar.header("Controls")

        dim_options = [f"{idx}: {st.session_state.atlas.get(idx, [{}])[0].get('label', 'N/A')}"
                       for idx in sorted(st.session_state.atlas.keys())]
        selected_dim_str = st.sidebar.selectbox("Search for a dimension", options=dim_options)
        selected_dim = int(selected_dim_str.split(':')[0])

        # 2. Display k images
        k_images = st.sidebar.slider("Number of images to display", 1, 20, 10)

        # 3. Toggle for patch visualization
        show_patches = st.sidebar.toggle("Highlight Activating Patches", value=True)

        st.sidebar.header("Edit Feature Label")
        current_label = st.session_state.atlas.get(selected_dim, [{}])[0].get('label', '')

        new_label = st.sidebar.text_input(
            "New Label",
            value=current_label,
            key=f"label_editor_{selected_dim}"
        )

        if st.sidebar.button("Update Label"):
            if new_label:
                # Update the label for all entries of this dimension in the session state
                for item in st.session_state.atlas[selected_dim]:
                    item['label'] = new_label

                # Save the entire updated atlas back to the file
                save_atlas_npz(ATLAS_PATH, st.session_state.atlas)

                st.cache_data.clear()
                st.sidebar.success(f"Label for dimension {selected_dim} updated!")
                st.rerun()
            else:
                st.sidebar.error("Label cannot be empty.")

        st.header(f"Dimension {selected_dim_str}")
        top_items = st.session_state.atlas.get(selected_dim, [])[:k_images]

        feat_extractor_info = {'grid_size': (14, 14)} # Todo: hardcoded, make param

        cols = st.columns(5)
        for i, item in enumerate(top_items):
            with cols[i % 5]:
                score = item.get('score', 0)
                st.caption(f"Score: {score:.2f} | Patches: {len(item['active_patch_indices'])}")

                if show_patches:
                    highlighted_img = create_highlighted_image(f"{datadir}/{item['image_path']}", item['active_patch_indices'], feat_extractor_info)
                    if highlighted_img:
                        st.image(highlighted_img, width='stretch')
                    else:
                        st.error("Image not found.")
                else:
                    st.image(f"{datadir}/{item['image_path']}", width='stretch')

    elif app_mode == "Cluster Explorer":
        st.header("Explore Feature Clusters")

        cluster_ids = sorted(st.session_state.clusters.keys(), key=int)
        selected_cluster_id = st.sidebar.selectbox("Select a Cluster", options=cluster_ids)

        st.subheader(f"Features in Cluster {selected_cluster_id}")
        st.write("Showing the top activating image for each feature in this cluster.")

        dims_in_cluster = st.session_state.clusters[selected_cluster_id]

        cols = st.columns(5)
        for i, dim_idx in enumerate(dims_in_cluster):
            with cols[i % 5]:
                top_item = st.session_state.atlas.get(dim_idx, [None])[0]
                if top_item:
                    st.markdown(f"**Dim: {dim_idx}**")
                    st.image(f"{datadir}/{top_item['image_path']}", width='stretch')

if __name__ == "__main__":
    args = parser.parse_args()
    main(dir=args.dir, datadir=args.data)
