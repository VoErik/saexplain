import streamlit as st
import os
import utils
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", page_title="Activation Analysis")

if 'current_sae' not in st.session_state:
    st.warning("Please select an SAE model on the 'Home' page first.")
    st.stop()

st.header("📊 Feature Activation Analysis")

sae_dir = st.session_state.current_sae
base_dir = st.session_state.base_dir
dir_path = os.path.join(base_dir, sae_dir)

df = utils.load_activation_stats(dir_path)

if df is None:
    st.error("Failed to load or process activation data.")
    st.stop()

total_features = len(df)
dead_features_mask = (df['frequency'] == 0)
num_dead = dead_features_mask.sum()
percent_dead = 100 * num_dead / total_features

st.metric(label="Total Features", value=f"{total_features}")
st.metric(label="Dead Features", value=f"{num_dead} ({percent_dead:.2f}%)")

active_df = df[~dead_features_mask].copy()

if active_df.empty:
    st.warning("No living features found. All features are dead.")
    st.stop()

epsilon = 1e-9
active_df['log10_frequency'] = np.log10(active_df['frequency'] + epsilon)
active_df['log10_mean_activation'] = np.log10(active_df['mean_activation'] + epsilon)

st.subheader("Histogram of Log10 Firing Frequencies")
fig_hist = px.histogram(
    active_df, 
    x='log10_frequency', 
    nbins=100,
    title="Log10 Firing Frequency Distribution (Living Features)"
)
fig_hist.update_layout(xaxis_title="Log10 Activation Frequency", yaxis_title="# Features")
st.plotly_chart(fig_hist, width="stretch")


st.subheader("Mean Activation vs. Firing Frequency")
st.info("Click and drag on the plot (using the 'Box Select' or 'Lasso Select' tools in the top-right) to select features.")

fig_scatter = px.scatter(
    active_df, 
    x='log10_frequency', 
    y='log10_mean_activation', 
    hover_name='feature_id',
    custom_data=['feature_id'], 
    title="Activation Frequency vs. Mean Magnitude (Living Features)",
    opacity=0.5
)
fig_scatter.update_layout(
    xaxis_title="Log10 Activation Frequency", 
    yaxis_title="Log10 Mean Activation",
    dragmode='select'
)

fig_scatter.update_traces(
    selected=dict(marker=dict(size=12)),
    selectedpoints=None
)

CHART_KEY = "analysis_plot_selection"
st.plotly_chart(
    fig_scatter,
    key=CHART_KEY,
    on_select="rerun",
    selection_mode="box",
    width="stretch"
)

if 'selection_cart' not in st.session_state:
    st.session_state.selection_cart = []

if CHART_KEY in st.session_state:
    selection_state = st.session_state[CHART_KEY]

    selected_points = []

    if isinstance(selection_state, dict):
        for key in ["selection", "select", "selected"]:
            sel = selection_state.get(key)
            if isinstance(sel, dict) and "points" in sel:
                selected_points = sel["points"]
                break

    if selected_points:
        selected_features = [p['customdata'][0] for p in selected_points]

        st.session_state.selection_cart = selected_features

        for key in ["selection", "select", "selected"]:
            if key in st.session_state[CHART_KEY]:
                st.session_state[CHART_KEY][key] = None

        st.rerun()


if st.session_state.selection_cart:

    selected_features_to_display = st.session_state.selection_cart
    
    st.header(f"Inspecting {len(selected_features_to_display)} Selected Features")

    if st.button("Clear Selection"):
        st.session_state.selection_cart = []
        st.rerun()

    atlas = st.session_state.atlas
    data_path = st.session_state.data_path
    img_size = (224, 224)
    feat_extractor_info = {'grid_size': (14, 14)}

    for dim_idx in selected_features_to_display:
        st.markdown("---")
        st.markdown(f"**Feature {dim_idx}**")

        if st.button(f"Go to Feature {dim_idx} (Explorer)", key=f"nav_btn_{dim_idx}"):
            st.session_state.feature_selector_state = f"Feature {dim_idx}"
            st.switch_page("pages/1_Single_Feature_Explorer.py")

        top_items = atlas.get(dim_idx, [])
        if not top_items:
            st.warning("No atlas images found for this feature.")
            continue

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
else:
    st.info("Select features on the plot above to inspect them here.")

st.sidebar.header("Explore Single Feature")
st.sidebar.info("Select on the plot or choose here.")

atlas_feature_ids = [f"Feature {k}" for k in st.session_state.atlas.keys() if len(st.session_state.atlas[k]) > 0]

if atlas_feature_ids:
    selected_feature_id = st.sidebar.selectbox(
        "Select a Feature ID",
        options=atlas_feature_ids,
        key="analysis_plot_selection"
    )

    if st.sidebar.button("Go to selected feature"):
        st.session_state.feature_selector_state = selected_feature_id
        st.switch_page("pages/1_Single_Feature_Explorer.py")