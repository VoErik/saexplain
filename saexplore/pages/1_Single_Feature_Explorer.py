import streamlit as st
from PIL import Image
import utils

st.set_page_config(layout="wide", page_title="Feature Explorer")


if 'atlas' not in st.session_state:
    st.warning("Please select an SAE model on the 'Home' page first.")
    st.stop()


atlas = st.session_state.atlas
data_path = st.session_state.data_path

st.sidebar.header("Explorer Controls")

all_dim_keys = sorted(atlas.keys())
dim_keys = [k for k in all_dim_keys if len(atlas[k]) > 0]

if not dim_keys:
    st.warning("The selected SAE Atlas contains no features with activating images.")
    st.stop()
    
dim_options = [f"Feature {idx}" for idx in dim_keys]

selected_option = st.sidebar.selectbox(
    "Search feature",
    options=dim_options,
    key="feature_selector_state" 
)

selected_dim = int(selected_option.split(" ")[1])

k_images = st.sidebar.slider("Number of images", 1, 5, 10, key="k_slider")

st.sidebar.markdown("---")
st.sidebar.header("Patch Visualization")
viz_type = st.sidebar.radio(
    "Display Mode", 
    ["Dimmed Mask", "Binary Mask", "Heatmap", "Original"], 
    horizontal=True,
    key="viz_type"
)

st.header("Single Feature Explorer")
st.subheader(f"Feature {selected_dim}: Top {k_images} Activating Images")

top_items = atlas.get(selected_dim, [])[:k_images]

feat_extractor_info = {'grid_size': (14, 14)} # TODO: put this stuff in explorer.yaml config
img_size = (224, 224)

cols = st.columns(5)
for i, item in enumerate(top_items):
    with cols[i % 5]:
        active_indices = item['active_patch_indices']
        active_values = item['active_values']
        active_patches_list = list(zip(active_indices, active_values))
        score = sum(active_values)
        
        st.caption(f"Score: {score:.2f} | Patches: {len(active_indices)}")
        
        full_img_path = item['image_path']

        try:
            image_to_show = None
            if viz_type == "Original":
                image_to_show = Image.open(full_img_path)
                image_to_show = image_to_show.resize(size=img_size)
            
            else:
                if viz_type == "Dimmed Mask":
                    image_to_show = utils.create_dimmed_mask_image(
                        full_img_path, active_indices, feat_extractor_info, img_size=img_size
                    )
                elif viz_type == "Binary Mask":
                    image_to_show = utils.create_binary_mask_image(
                        full_img_path, active_indices, feat_extractor_info, img_size=img_size
                    )
                elif viz_type == "Heatmap":
                    image_to_show = utils.create_heatmap_image(
                        full_img_path, active_patches_list, feat_extractor_info, img_size=img_size
                    )

            if image_to_show:
                st.image(image_to_show, width='stretch')
                
        except FileNotFoundError:
            st.error(f"Img not found: {item['image_path']}")
        except Exception as e:
            st.error(f"Error loading image: {e}")