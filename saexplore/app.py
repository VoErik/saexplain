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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Todo: fix this ugly thing

import utils
from src.models.sae import SAE
from src.utils.load_backbone import load_encoder


def main(dir_path, data_path):
    st.set_page_config(layout="wide")
    st.title("🔬 Interactive SAE Atlas Explorer")

    # --- File Paths --- # Todo: make this a dropdown selection in app / make folder containing those drop down
    ATLAS_PATH = f"{dir_path}/sae_atlas.npz"
    CLUSTERS_PATH = f"{dir_path}/sae_clusters.json"
    SAE_MODEL_PATH = f"{dir_path}/sae_model"
    ACTIVATIONS_PATH = f"{dir_path}/all_activations.pt"
    UMAP_PATH = f"{dir_path}/umap.npy"


    # --- Load Data into Session State ---
    if 'atlas' not in st.session_state:
        atlas, clusters = utils.load_data(ATLAS_PATH, CLUSTERS_PATH)
        if atlas is None or clusters is None:
            st.stop()
        st.session_state.atlas = atlas
        st.session_state.clusters = clusters

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Explorer",
        [
            "Single Feature Explorer",
            "Cluster Explorer",
            "UMAP",
            "Inference Mode",
            "Activation Analysis"
        ],
        key="app_mode"
    )

    # --- Single Feature Explorer ---
    if st.session_state.app_mode == "Single Feature Explorer":
        st.sidebar.header("Controls")

        # --- Dimension Selection & Navigation ---
        dim_keys = sorted(st.session_state.atlas.keys())
        dim_options = [
            f"{idx}: {st.session_state.atlas[idx][0]['label']}" for idx in dim_keys
        ]

        default_index = 0
        if 'navigate_to_dim' in st.session_state:
            try:
                default_index = dim_keys.index(st.session_state.navigate_to_dim)
            except ValueError:
                default_index = 0 # default
            del st.session_state.navigate_to_dim

        selected_dim_str = st.sidebar.selectbox(
            "Search dimension",
            options=dim_options,
            index=default_index
        )
        selected_dim = int(selected_dim_str.split(':')[0])

        k_images = st.sidebar.slider("Number of images to display", 1, 20, 10)

        # --- Visualization Mode Selector ---
        st.sidebar.markdown("---")
        st.sidebar.header("Patch Visualization")
        viz_type = st.sidebar.radio(
            "Display Mode",
            [
                "Dimmed Mask",
                "Binary Mask",
                "Heatmap",
                "Original"
            ],
            horizontal=True,
        )

        # --- Label Editing ---
        st.sidebar.markdown("---")
        st.sidebar.header("Edit Feature Label")
        current_label = st.session_state.atlas[selected_dim][0]['label']
        new_label = st.sidebar.text_input(
            "New Label",
            value=current_label,
            key=f"label_editor_{selected_dim}"
        )

        if st.sidebar.button("Update Label"):
            if new_label:
                for item in st.session_state.atlas[selected_dim]:
                    item['label'] = new_label

                utils.save_atlas_npz(ATLAS_PATH, st.session_state.atlas)

                st.cache_data.clear()
                st.sidebar.success("Label updated!")
                st.rerun()
            else:
                st.sidebar.error("Label cannot be empty.")

        # --- Main Page Display ---
        st.header(f"Dimension {selected_dim_str}")
        top_items = st.session_state.atlas.get(selected_dim, [])[:k_images]

        # TODO: This should be made configurable based on feature extractor model
        feat_extractor_info = {'grid_size': (14, 14)}

        cols = st.columns(5)
        for i, item in enumerate(top_items):
            with cols[i % 5]:
                score = item.get('score', 0)
                active_patches_list = item.get('active_patch_indices', item.get('active_patches', []))
                st.caption(f"Score: {score:.2f} | Patches: {len(active_patches_list)}")

                full_img_path = f"{data_path}/{item['image_path']}"

                image_to_show = None
                try:
                    if viz_type == "Original":
                        image_to_show = Image.open(full_img_path)

                    elif viz_type == "Dimmed Mask":
                        indices = item.get('active_patch_indices', [p[0] for p in item.get('active_patches', [])])
                        image_to_show = utils.create_dimmed_mask_image(
                            full_img_path, indices, feat_extractor_info
                        )

                    elif viz_type == "Binary Mask":
                        indices = item.get('active_patch_indices', [p[0] for p in item.get('active_patches', [])])
                        image_to_show = utils.create_binary_mask_image(
                            full_img_path, indices, feat_extractor_info
                        )

                    elif viz_type == "Heatmap":
                        if 'active_patches' in item:
                            image_to_show = utils.create_heatmap_image(
                                full_img_path, item['active_patches'], feat_extractor_info
                            )
                        else:
                            st.warning("Heatmap requires activation values.")
                            indices = item.get('active_patch_indices', [])
                            image_to_show = utils.create_dimmed_mask_image(
                                full_img_path, indices, feat_extractor_info
                            )

                    if image_to_show:
                        st.image(image_to_show, width='stretch')
                    else:
                        st.error("Image not found.")
                except FileNotFoundError:
                    st.error(f"File not found: {item['image_path']}")

    # --- Cluster Explorer ---
    elif st.session_state.app_mode == "Cluster Explorer":
        st.header("Explore Feature Clusters")
        cluster_ids = sorted(st.session_state.clusters.keys(), key=int)
        selected_cluster_id = st.sidebar.selectbox("Select a Cluster", options=cluster_ids)

        dims_in_cluster = st.session_state.clusters[selected_cluster_id]
        st.subheader(f"{len(dims_in_cluster)} Features in Cluster {selected_cluster_id}")

        for dim_idx in dims_in_cluster:
            st.markdown(f"---")
            top_item = st.session_state.atlas.get(dim_idx, [{}])[0]
            st.markdown(f"**Dimension {dim_idx}: {top_item.get('label', 'N/A')}**")

            if st.button(f"Go to Feature {dim_idx}", key=f"explore_{dim_idx}"):
                st.session_state.navigate_to_dim = dim_idx
                st.session_state.app_mode = "Single Feature Explorer"
                st.rerun()

            cols = st.columns(5)
            for i, item in enumerate(st.session_state.atlas.get(dim_idx, [])[:5]):
                with cols[i]:
                    st.image(f"{data_path}/{item['image_path']}", width='stretch')

    # --- UMAP & Frequency Explorer ---
    elif st.session_state.app_mode == "UMAP":
        st.header("🗺️ UMAP Visualization of SAE Features")

        umap_embeddings = utils.load_umap_embeddings(UMAP_PATH)
        frequencies = utils.calculate_activation_frequencies(ACTIVATIONS_PATH)

        if umap_embeddings is None:
            st.error(f"UMAP embeddings file not found at: {UMAP_PATH}")
            st.info("Please run the `precompute_umap.py` script first.")
            st.stop()

        if frequencies is None:
            st.error(f"Activation file not found at: {ACTIVATIONS_PATH}")
            st.info("Please run the `get_all_activations` function first.")
            st.stop()

        # --- Sidebar Controls ---
        st.sidebar.header("UMAP Controls")
        log_color_scale = st.sidebar.checkbox("Use Log Scale for Color", value=True)

        df = pd.DataFrame({
            'x': umap_embeddings[:, 0],
            'y': umap_embeddings[:, 1],
            'frequency': frequencies,
            'feature_id': np.arange(len(frequencies))
        })

        if log_color_scale:
            df['log_frequency'] = np.log10(df['frequency'] + 1e-5)
            color_col = 'log_frequency'
            color_label = "Frequency (log %)"
        else:
            color_col = 'frequency'
            color_label = "Frequency (%)"

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_col,
            hover_data={
                'x': False, # Hide x coordinate from hover
                'y': False, # Hide y coordinate from hover
                'feature_id': True, # Show the feature ID
                'frequency': ':.3f%', # Show frequency, formatted as a percentage
            },
            title="SAE Features Clustered by UMAP, Colored by Activation Frequency"
        )

        fig.update_layout(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            coloraxis_colorbar_title_text=color_label
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **How to Read This Chart:**
        * **Points** that are close together represent SAE features that are semantically similar (based on their decoder weights).
        * **Color** indicates how often a feature activates across the dataset.
        """)

    elif st.session_state.app_mode == "Activation Analysis":
        st.header("📊 Feature Activation Analysis")
        st.write("This plot shows the relationship between how often a feature activates (frequency) and how strongly it activates on average when it does (magnitude).")

        frequencies = utils.calculate_activation_frequencies(ACTIVATIONS_PATH)
        mean_activations = utils.calculate_mean_activations(ACTIVATIONS_PATH)

        if frequencies is None or mean_activations is None:
            st.error(f"Activation file not found at: {ACTIVATIONS_PATH}")
            st.info("Please ensure you have run the `get_all_activations` function.")
            st.stop()

        df = pd.DataFrame({
            'feature_id': np.arange(len(frequencies)),
            'frequency': frequencies,
            'mean_activation': mean_activations
        })

        # Filter out dead features
        active_df = df[df['frequency'] > 0].copy()

        active_df['log10_frequency'] = np.log10(active_df['frequency'])
        active_df['log10_mean_activation'] = np.log10(active_df['mean_activation'])

        fig = px.scatter(
            active_df,
            x='log10_frequency',
            y='log10_mean_activation',
            hover_name='feature_id',
            hover_data={
                'frequency': ':.3f%',
                'mean_activation': ':.3f',
                'log10_frequency': False,
                'log10_mean_activation': False
            },
            labels={
                'log10_frequency': 'log10 Activation Frequency (%)',
                'log10_mean_activation': 'log10 Mean Activation Value'
            },
            title="Activation Frequency vs. Mean Activation Magnitude"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Displaying {len(active_df)} out of {len(df)} total features "
                f"(dead features with zero frequency are excluded from the log plot).")

    elif st.session_state.app_mode == "Inference Mode":
        st.header("📸 Inference")
        st.write("Upload an image to see which SAE features activate most strongly on it.")

        @st.cache_resource
        def load_models_for_inference():
            """Loads the feature extractor and SAE model for inference."""
            feature_extractor, prep = load_encoder(
                "dinov3_vitl16", "../dinov3", "cuda"
            ) # Todo: make this param -> should be read from some info.txt file in analysis dir
            sae_model = SAE.load_model(SAE_MODEL_PATH)
            return feature_extractor, sae_model, prep

        feature_extractor, sae_model, prep = load_models_for_inference()

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing image..."):
                top_indices, top_scores = utils.run_inference_on_image(
                    image, feature_extractor, sae_model, prep, "cuda"
                )

            st.subheader("Top Activating Features:")

            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                label = st.session_state.atlas[idx][0]['label']

                st.markdown("---")
                st.markdown(f"**{i+1}. Feature {idx}: {label}** (Score: {score:.2f})")
                st.caption(f"Top 5 examples for {label}:")

                top_examples_from_atlas = st.session_state.atlas[idx][:5]

                example_cols = st.columns(5)

                for col, item in zip(example_cols, top_examples_from_atlas):
                    with col:
                        full_img_path = f"{data_path}/{item['image_path']}"
                        try:
                            st.image(full_img_path)
                        except FileNotFoundError:
                            st.error("Img not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./analysis")
    parser.add_argument("--data", type=str, default=".")
    args = parser.parse_args()
    main(dir_path=args.dir, data_path=args.data)
