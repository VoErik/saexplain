import streamlit as st
import os

import argparse
import utils

import streamlit as st
import os
import argparse
import utils

def main(base_dir: str):
    st.set_page_config(
        layout="wide",
        page_title="SAE Atlas Explorer",
        page_icon="🔬"
    )
    st.title("🔬 SAE Atlas Explorer")
    
    st.sidebar.title("Configuration")
    sae_options = utils.get_sae_dirs(base_dir)
    
    if not sae_options:
        st.error(f"No valid SAE model directories found in: {base_dir}")
        st.info("A valid directory must contain an 'atlas.json' file.")
        st.stop()

    def on_sae_change():
        keys_to_reset = ['atlas', 'clusters', 'current_sae']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]


    selected_sae_dir = st.sidebar.selectbox(
        "Select SAE Model", 
        options=sae_options, 
        key="sae_selector", 
        on_change=on_sae_change
    )
    
    dir_path = os.path.join(base_dir, selected_sae_dir)
    ATLAS_PATH = os.path.join(dir_path, "atlas.json")
    CLUSTERS_PATH = os.path.join(dir_path, "sae_clusters_top100_pearson_common.json")

    if 'atlas' not in st.session_state or st.session_state.get('current_sae') != selected_sae_dir:
        with st.spinner(f"Loading data for {selected_sae_dir}..."):
            atlas, clusters = utils.load_data(ATLAS_PATH, CLUSTERS_PATH)
            if atlas is None:
                st.error(f"Failed to load atlas for {selected_sae_dir}. Check logs.")
                st.stop()
            
            st.session_state.atlas = atlas
            st.session_state.clusters = clusters
            st.session_state.current_sae = selected_sae_dir

    st.header("Welcome!")
    st.info(f"Loaded **{selected_sae_dir}** with **{len(st.session_state.atlas)}** features.")
    st.markdown("Select a view from the sidebar to begin exploring.")
    st.sidebar.success("Model loaded. Ready to explore!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./assets/configs/explorer_config.yaml", help="Path to config file.")
    args = parser.parse_args()
    
    if args.config:
        import yaml
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError("You need to provide a configuration file.")

    st.session_state.data_path = config["data_dir"]
    st.session_state.base_dir = config["sae_dir"]
    
    main(base_dir=config["sae_dir"])