import streamlit as st
from PIL import Image
import os
import utils

st.set_page_config(layout="wide", page_title="Cluster Explorer")

if 'atlas' not in st.session_state:
    st.warning("Please select an SAE model on the 'Home' page first.")
    st.stop()

atlas = st.session_state.atlas
clusters = st.session_state.clusters
data_path = st.session_state.data_path

st.header("Explore Feature Clusters")

if clusters is None:
    st.error("No cluster data found.")
    st.info("Ensure 'sae_clusters.json' exists in your SAE directory.")
    st.stop()

st.sidebar.header("Cluster Controls")

cluster_ids = sorted(clusters.keys(), key=int)
selected_cluster_id = st.sidebar.selectbox(
    "Select a Cluster", 
    options=cluster_ids,
    format_func=lambda x: f"Cluster {x}"
)

dims_in_cluster = clusters[selected_cluster_id]
st.subheader(f"{len(dims_in_cluster)} Features in Cluster {selected_cluster_id}")

img_size = (224, 224) # TODO: make this stuff part of explorer.yaml config
feat_extractor_info = {'grid_size': (14, 14)}

for dim_idx in dims_in_cluster:
    st.markdown("---")
    
    top_items = atlas.get(dim_idx, [])
    
    if not top_items:
        st.markdown(f"**Feature {dim_idx}** (No activating images found in atlas)")
        continue

    st.markdown(f"**Feature {dim_idx}**")

    if st.button(f"Go to Feature {dim_idx}", key=f"nav_btn_{dim_idx}"):
        st.session_state.feature_selector_state = f"Feature {dim_idx}"
        
        st.switch_page("pages/1_Single_Feature_Explorer.py")

    cols = st.columns(5)
    for i, item in enumerate(top_items[:5]):
        with cols[i % 5]:
            full_img_path = item['image_path']
            try:
                image_to_show = utils.create_dimmed_mask_image(
                    full_img_path, 
                    item['active_patch_indices'], 
                    feat_extractor_info, 
                    img_size=img_size
                )
                if image_to_show:
                    st.image(image_to_show, width='stretch')
                else:
                    st.error("Img not found.")
            except FileNotFoundError:
                st.error(f"Img not found: {item['image_path']}")
            except Exception as e:
                st.error(f"Error: {e}")