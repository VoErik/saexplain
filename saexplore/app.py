# app.py
import streamlit as st
import argparse
import torch
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import os
import sys

# Allows the script to find modules in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from src.models.sae import SAE
from src.utils.load_backbone import load_encoder

def get_sae_dirs(base_dir):
    """Finds all valid SAE directories within the base analysis directory."""
    sae_dirs = []
    if not os.path.isdir(base_dir):
        return []
    for dir_name in os.listdir(base_dir):
        path = os.path.join(base_dir, dir_name)
        if os.path.isdir(path) and 'sae_'in path:
            sae_dirs.append(dir_name)
    return sae_dirs

def main(base_dir, data_path):
    st.set_page_config(layout="wide")
    st.title("🔬 Interactive SAE Atlas Explorer")

    # --- 💡 EXPERT NAVIGATION HANDLER ---
    # This logic runs at the start of every script run. It checks for navigation
    # requests set by buttons and updates the main view state *before* any widgets
    # are rendered, preventing the StreamlitAPIException.
    if 'navigate_to_view' in st.session_state:
        st.session_state.app_mode = st.session_state.navigate_to_view
        del st.session_state.navigate_to_view  # Clean up the request

    # --- SAE Model Selection ---
    st.sidebar.title("Configuration")
    sae_options = get_sae_dirs(base_dir)
    if not sae_options:
        st.error(f"No valid SAE model directories found in: {base_dir}")
        st.stop()

    # When the SAE model is changed, we must reset the navigation state
    def on_sae_change():
        keys_to_delete = ['atlas', 'clusters', 'current_sae', 'navigate_to_dim']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.app_mode = "Single Feature Explorer"

    selected_sae_dir = st.sidebar.selectbox(
        "Select SAE Model", options=sae_options, key="sae_selector", on_change=on_sae_change
    )
    dir_path = os.path.join(base_dir, selected_sae_dir)

    # --- File Paths ---
    ATLAS_PATH = f"{dir_path}/sae_atlas.npz"
    CLUSTERS_PATH = f"{dir_path}/sae_clusters.json"
    SAE_MODEL_PATH = f"{dir_path}"
    ACTIVATIONS_PATH = f"{dir_path}/all_activations.pt"
    UMAP_PATH = f"{dir_path}/umap.npy"
    img_size: tuple[int, int] = (224,224)

    # --- Load Data into Session State ---
    if 'atlas' not in st.session_state or st.session_state.get('current_sae') != selected_sae_dir:
        with st.spinner(f"Loading data for {selected_sae_dir}..."):
            st.session_state.current_sae = selected_sae_dir
            atlas, clusters = utils.load_data(ATLAS_PATH, CLUSTERS_PATH)
            if atlas is None or clusters is None:
                st.error(f"Failed to load atlas or clusters for {selected_sae_dir}. Check file paths.")
                st.stop()
            st.session_state.atlas = atlas
            st.session_state.clusters = clusters

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    st.sidebar.radio(
        "Choose Explorer",
        ["Single Feature Explorer", "Cluster Explorer", "UMAP", "Inference Mode", "Activation Analysis"],
        key="app_mode"
    )

    # --- Single Feature Explorer ---
    if st.session_state.app_mode == "Single Feature Explorer":
        st.sidebar.header("Controls")
        all_dim_keys = sorted(st.session_state.atlas.keys())
        dim_keys = [k for k in all_dim_keys if len(st.session_state.atlas[k]) > 1]

        if not dim_keys:
            st.warning("The selected SAE Atlas contains no features with activating images.")
            st.stop()
        dim_options = [f"{idx}: {st.session_state.atlas[idx][0].get('label', str(idx))}" for idx in dim_keys]

        # --- CORRECTED STATE LOGIC FOR FEATURE SELECTOR ---
        # 1. Check for a one-time navigation request.
        if 'navigate_to_dim' in st.session_state:
            try:
                # If we have a request, set the selector's index state and consume the request.
                st.session_state.feature_selector_index = dim_keys.index(st.session_state.navigate_to_dim)
                del st.session_state.navigate_to_dim
            except (ValueError, IndexError):
                st.session_state.feature_selector_index = 0
        # 2. If no request, ensure the index state exists (for the first run).
        elif 'feature_selector_index' not in st.session_state:
            st.session_state.feature_selector_index = 0

        # 3. Create the widget, controlled by our persistent index state.
        #    The `on_change` callback updates our state variable when the user selects a new option.
        def on_feature_select_change():
            st.session_state.feature_selector_index = st.session_state.feature_selector_widget

        selected_index = st.sidebar.selectbox(
            "Search feature",
            options=range(len(dim_options)),
            format_func=lambda i: dim_options[i],
            index=st.session_state.feature_selector_index,
            key='feature_selector_widget',
            on_change=on_feature_select_change
        )

        # Get the dimension from the correctly persisted index
        selected_dim_str = dim_options[selected_index]
        selected_dim = int(selected_dim_str.split(':')[0])

        k_images = st.sidebar.slider("Number of images to display", 1, 20, 10)
        st.sidebar.markdown("---"); st.sidebar.header("Patch Visualization")
        viz_type = st.sidebar.radio("Display Mode", ["Dimmed Mask", "Binary Mask", "Heatmap", "Original"], horizontal=True)

        st.sidebar.markdown("---"); st.sidebar.header("Edit Feature Label")
        current_label = st.session_state.atlas[selected_dim][0].get('label', str(selected_dim))
        new_label = st.sidebar.text_input("New Label", value=current_label, key=f"label_editor_{selected_dim}")

        if st.sidebar.button("Update Label"):
            if new_label:
                for item in st.session_state.atlas[selected_dim]: item['label'] = new_label
                utils.save_atlas_npz(ATLAS_PATH, st.session_state.atlas)
                st.sidebar.success("Label updated!"); st.rerun()
            else:
                st.sidebar.error("Label cannot be empty.")

        # --- Main Page Display ---
        st.header(f"Feature {selected_dim_str}")
        top_items = st.session_state.atlas.get(selected_dim, [])[:k_images]
        feat_extractor_info = {'grid_size': (14, 14)}
        cols = st.columns(5)
        for i, item in enumerate(top_items):
            with cols[i % 5]:
                score = item.get('score', 0)
                active_patches_list = item.get('active_patches', [])
                st.caption(f"Score: {score:.2f} | Patches: {len(active_patches_list)}")
                full_img_path = os.path.join(data_path, item['image_path'])
                try:
                    image_to_show = None
                    if viz_type == "Original":
                        image_to_show = Image.open(full_img_path)
                        image_to_show = image_to_show.resize(size=img_size)
                    else:
                        indices = [p[0] for p in active_patches_list]
                        if viz_type == "Dimmed Mask": image_to_show = utils.create_dimmed_mask_image(
                            full_img_path, indices, feat_extractor_info, img_size=img_size
                        )
                        elif viz_type == "Binary Mask": image_to_show = utils.create_binary_mask_image(
                            full_img_path, indices, feat_extractor_info, img_size=img_size
                        )
                        elif viz_type == "Heatmap":
                            if active_patches_list: image_to_show = utils.create_heatmap_image(
                                full_img_path, active_patches_list, feat_extractor_info, img_size=img_size
                            )
                            else:
                                st.warning("Heatmap data NA.");
                                image_to_show = Image.open(full_img_path)
                                image_to_show = image_to_show.resize(size=img_size)
                    if image_to_show: st.image(image_to_show, width='stretch')
                except FileNotFoundError: st.error(f"Image not found.")

    # --- Cluster Explorer ---
    elif st.session_state.app_mode == "Cluster Explorer":
        st.header("Explore Feature Clusters")
        cluster_ids = sorted(st.session_state.clusters.keys(), key=int)
        selected_cluster_id = st.sidebar.selectbox("Select a Cluster", options=cluster_ids)
        dims_in_cluster = st.session_state.clusters[selected_cluster_id]
        st.subheader(f"{len(dims_in_cluster)} Features in Cluster {selected_cluster_id}")
        for dim_idx in dims_in_cluster:
            st.markdown("---")
            top_item = st.session_state.atlas.get(dim_idx, [{}])[0]
            st.markdown(f"**Feature {dim_idx}: {top_item.get('label', 'N/A')}**")
            if st.button(f"Go to Feature {dim_idx}", key=f"explore_{dim_idx}"):
                st.session_state.navigate_to_dim = dim_idx
                st.session_state.navigate_to_view = "Single Feature Explorer"
                st.rerun()
            cols = st.columns(5)
            for i, item in enumerate(st.session_state.atlas.get(dim_idx, [])[:5]):
                with cols[i]:
                    full_img_path = os.path.join(data_path, item['image_path'])
                    try:
                        image_to_show = Image.open(full_img_path).resize(size=img_size)
                        st.image(image_to_show, width='stretch')
                    except FileNotFoundError: st.error("Img not found.")

    # --- Activation Analysis ---
    elif st.session_state.app_mode == "Activation Analysis":
        st.header("📊 Feature Activation Analysis")
        frequencies = utils.calculate_activation_frequencies(ACTIVATIONS_PATH)
        mean_activations = utils.calculate_mean_activations(ACTIVATIONS_PATH)

        if frequencies is None or mean_activations is None:
            st.error("Activation data not found.")
            st.stop()

        df = pd.DataFrame({'feature_id': np.arange(len(frequencies)), 'frequency': frequencies, 'mean_activation': mean_activations})
        active_df = df[df['frequency'] > 0].copy()

        st.sidebar.header("Explore from Plot")
        feature_options = active_df['feature_id'].astype(str).tolist()
        selected_feature = st.sidebar.selectbox("Select a Feature ID to explore", options=feature_options)
        if st.sidebar.button("Go to selected feature"):
            st.session_state.navigate_to_dim = int(selected_feature)
            st.session_state.navigate_to_view = "Single Feature Explorer"
            st.rerun()

        if not active_df.empty:
            active_df['log10_frequency'] = np.log10(active_df['frequency'])
            active_df['log10_mean_activation'] = np.log10(active_df['mean_activation'])
            fig = px.scatter(
                active_df, x='log10_frequency', y='log10_mean_activation', hover_name='feature_id',
                hover_data={'frequency': ':.3f%', 'mean_activation': ':.3f'},
                labels={'log10_frequency': 'log10 Freq (%)', 'log10_mean_activation': 'log10 Mean Activation'},
                title="Activation Frequency vs. Mean Magnitude"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Displaying {len(active_df)} of {len(df)} features (dead features excluded).")
        else:
            st.warning("No active features found to plot.")

    # --- Other Views ---
    elif st.session_state.app_mode == "UMAP":
        st.header("🗺️ UMAP Visualization of SAE Features")
        umap_embeddings = utils.load_umap_embeddings(UMAP_PATH)
        frequencies = utils.calculate_activation_frequencies(ACTIVATIONS_PATH)
        if umap_embeddings is None or frequencies is None:
            st.error("Could not load UMAP or activation data.")
            st.stop()
        df = pd.DataFrame({'x': umap_embeddings[:, 0], 'y': umap_embeddings[:, 1], 'frequency': frequencies, 'feature_id': np.arange(len(frequencies))})
        df['log_frequency'] = np.log10(df['frequency'] + 1e-5)
        fig = px.scatter(df, x='x', y='y', color='log_frequency', hover_data={'feature_id': True, 'frequency': ':.3f%'}, title="SAE Features by UMAP")
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.app_mode == "Inference Mode":
        st.header("📸 Inference")
        @st.cache_resource
        def load_models_for_inference(sae_path):
            feature_extractor, prep = load_encoder("dinov3_vitl16", "../dinov3", "cuda")
            sae_model = SAE.load_model(sae_path)
            return feature_extractor, sae_model, prep
        feature_extractor, sae_model, prep = load_models_for_inference(SAE_MODEL_PATH)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width='stretch')
            with st.spinner("Analyzing..."):
                top_indices, top_scores = utils.run_inference_on_image(image, feature_extractor, sae_model, prep, "cuda")
            st.subheader("Top Activating Features:")
            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                label = st.session_state.atlas[idx][0].get('label', str(idx))
                st.markdown(f"--- \n **{i+1}. Feature {idx}: {label}** (Score: {score:.2f})")
                cols = st.columns(5)
                for col, item in zip(cols, st.session_state.atlas.get(idx, [])[:5]):
                    full_img_path = os.path.join(data_path, item['image_path'])
                    try: col.image(full_img_path)
                    except FileNotFoundError: col.error("Img missing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./analysis", help="Base directory for analysis files from different SAEs.")
    parser.add_argument("--data", type=str, default=".", help="Root directory for the dataset images.")
    args = parser.parse_args()
    main(base_dir=args.dir, data_path=args.data)
