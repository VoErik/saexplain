from typing import Literal

from src.models.sae import SAE
from src.models.sae.sae_utils import (
    build_sae_atlas,
    plot_top_k_images,
    plot_activation_histogram,
    plot_dendrogram_with_images,
    analyze_and_cluster_features,
    get_all_activations
)


def run_analysis(
        path: str,
        feature_extractor: str,
        extractor_path: str,
        device: str,
        dataset: str,
        output_path: str,
        aggregation: Literal['mean', 'binary_sum']  = "binary_sum",
        threshold: float = 0.1,
        k: int = 20
):
    mod = SAE.load_model(path, device=device)

    build_sae_atlas(
        feature_extractor=(feature_extractor, extractor_path),
        sae_model=mod,
        dataset=dataset,
        sae_dimensions=8192,
        k=k,
        device=device,
        output_path=f"{output_path}/sae_atlas",
        aggregation=aggregation,
        threshold=threshold,
        batch_size=32
    )


    plot_top_k_images(atlas_path=f"{output_path}/sae_atlas.npz", num_dims_to_plot=10, k_to_show=8, savedir=output_path)
    plot_activation_histogram(atlas_path=f"{output_path}/sae_atlas.npz", savedir=output_path)

    all_activations_tensor = get_all_activations(
         feature_extractor_name=(feature_extractor, extractor_path),
         sae_model=mod,
         dataset_name=dataset,
         device="cuda",
         batch_size=64
     )

    linkage_matrix, topk_indices = analyze_and_cluster_features(
        all_activations=all_activations_tensor,
        atlas_path=f"{output_path}/sae_atlas.npz",
        k=100,
        savedir=output_path,
        cluster_threshold=0.6
     )

    plot_dendrogram_with_images(
        linkage_matrix=linkage_matrix,
        top_k_indices=topk_indices,
        atlas_path=f"{output_path}/sae_atlas.npz",
        k_images=3,
        savedir=output_path
     )


if __name__ == "__main__":
    sae_path = "checkpoints/sae_fitzpatrick17k_topk_d_in_1024_d_sae_8192_act_topk_k64_auxcoeff_2.0"
    device = "cuda"
    extractor = "dinov3_vitl16"
    extractor_path = "../dinov3"
    output_path = "./analysis"
    dataset = "skincon_fitzpatrick17k"

    run_analysis(
        path=sae_path,
        feature_extractor=extractor,
        extractor_path=extractor_path,
        device="cuda",
        dataset=dataset,
        output_path=output_path,
        k=20,
        threshold=0.5
    )