import heapq
from typing import Dict, List, Any, Tuple, Literal

import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import torchvision

from src.dataloaders import ImageDataManager
from src.utils.load_backbone import load_encoder
import numpy as np
import umap


######################################
### INTERPRETABILITY FUNCTIONALITY ###
######################################

def aggregate_activations(
        activations: torch.Tensor,
        aggregation: Literal['mean', 'binary_sum'],
        threshold: float
) -> torch.Tensor:
    """
    Aggregates patch-level SAE activations to image-level scores.
    Expects a 2D tensor of shape (num_patches, sae_dimensions).
    """
    if aggregation == 'mean':
        # Aggregate across the patch dimension (dim=0)
        return torch.mean(activations, dim=0)
    elif aggregation == 'binary_sum':
        # Aggregate across the patch dimension (dim=0)
        return (activations > threshold).float().sum(dim=0)
    else:
        raise ValueError(f"Aggregation method '{aggregation}' is not supported.")

def build_sae_atlas(
        feature_extractor: Tuple[str, str],
        sae_model: nn.Module,
        dataset: str,
        sae_dimensions: int,
        k: int,
        device: str,
        output_path: str,
        batch_size: int = 64,
        aggregation: Literal['mean', 'binary_sum'] = 'binary_sum',
        threshold: float = 0.1,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Computes and saves a top-k SAE atlas.
    """
    feature_extractor, prep = load_encoder(feature_extractor[0], feature_extractor[1], device)
    feature_extractor.to(device).eval()
    sae_model.to(device).eval()

    dm = ImageDataManager(initialize=dataset, transform=prep)
    dataloader, _ = dm.get_dataloaders(dataset=dataset, batch_size=batch_size, test_size=0.0, shuffle=False)

    min_scores = torch.full((sae_dimensions,), -float('inf'), device=device)
    atlas_heaps = {i: [] for i in range(sae_dimensions)}
    tie_breaker_counter = 0

    with torch.inference_mode():
        for images, label_dicts in tqdm(dataloader, desc="Building SAE Atlas"):
            images = images.to(device)

            if "clip" in str(type(feature_extractor)).lower():
                feature_extractor.visual.output_tokens = True
                _ , patch_embeddings = feature_extractor.visual.forward(images)
            elif "dino" in str(type(feature_extractor)).lower():
                x = feature_extractor.forward_features(images)
                patch_embeddings = x["x_norm_patchtokens"]
            else:
                patch_embeddings = feature_extractor(images)

            activations = sae_model.encode(patch_embeddings) # Expected Shape: (B, Num_Patches, SAE_Dim)

            for i in range(images.shape[0]):
                img_path = label_dicts["img_path"][i]
                item_activations = activations[i]

                image_level_scores = aggregate_activations(
                    activations=item_activations, aggregation=aggregation, threshold=threshold
                )

                candidate_dims = torch.where(image_level_scores > min_scores)[0]

                for dim_idx_tensor in candidate_dims:
                    dim_idx = dim_idx_tensor.item()
                    score = image_level_scores[dim_idx].item()
                    heap = atlas_heaps[dim_idx]

                    patch_scores = item_activations[:, dim_idx]
                    active_indices_tensor = torch.where(patch_scores > threshold)[0]

                    active_patches_data = [
                        [idx.item(), patch_scores[idx].item()] for idx in active_indices_tensor
                    ]

                    new_entry = {
                        "score": score,
                        "sae_dimension": dim_idx,
                        "image_path": img_path,
                        "active_patches": active_patches_data,
                        "label": str(dim_idx)
                    }

                    heapq.heappush(heap, (score, tie_breaker_counter, new_entry))
                    tie_breaker_counter += 1

                    if len(heap) > k:
                        heapq.heappop(heap)

                    min_scores[dim_idx] = heap[0][0]

    final_atlas = {}
    for dim_idx, heap in tqdm(atlas_heaps.items(), desc="Finalizing Atlas"):
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
        final_atlas[dim_idx] = [item[2] for item in sorted_items]

    output_file_path = f"{output_path}.npz" if not output_path.endswith('.npz') else output_path
    np.savez_compressed(output_file_path, **{str(k): np.array(v, dtype=object) for k, v in final_atlas.items()})
    print(f"SAE atlas built and saved to {output_file_path}")

    return final_atlas


def rank_dimensions(atlas: Dict[int, List[Dict[str, Any]]]) -> List[int]:
    """Ranks SAE dimensions by the mean activation of their top-k images."""
    dim_importances = {}
    for dim_idx, top_items in atlas.items():
        if not top_items:
            continue
        mean_score = sum(item["score"] for item in top_items) / len(top_items)
        dim_importances[dim_idx] = mean_score

    sorted_dims = sorted(dim_importances, key=dim_importances.get, reverse=True)
    return sorted_dims


def plot_top_k_images(
        atlas_path: str,
        num_dims_to_plot: int,
        k_to_show: int = 5,
        savedir: str = "plots"
):
    """
    Plots top activating images from a saved SAE atlas npz file.
    """
    f = np.load(atlas_path, allow_pickle=True)
    atlas = {int(k): v.tolist() for k, v in f.items()}

    ranked_dims = rank_dimensions(atlas)
    dims_to_plot = ranked_dims[:num_dims_to_plot]

    k_in_atlas = len(next(iter(atlas.values())))
    k_to_show = min(k_to_show, k_in_atlas)

    fig, axes = plt.subplots(
        nrows=len(dims_to_plot),
        ncols=k_to_show,
        figsize=(k_to_show * 2.5, len(dims_to_plot) * 2.5),
        squeeze=False
    )

    for i, dim_idx in enumerate(dims_to_plot):
        top_items = atlas[dim_idx][:k_to_show]
        dim_label = top_items[0]['label']
        axes[i][0].set_ylabel(f"Dim {dim_label}", rotation=0, size='large', labelpad=45)

        for j, item in enumerate(top_items):
            score = item["score"]
            img_path = item["image_path"]
            ax = axes[i][j]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"Score: {score:.2f}")
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center')

            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.5, h_pad=1.5)
    plt.savefig(f'{savedir}/sae_topk_images.png', dpi=300)


def plot_activation_histogram(atlas_path: str, savedir: str = "plots"):
    """
    Visualizes the importance of all features.
    """
    f = np.load(atlas_path, allow_pickle=True)
    atlas = {int(k): v.tolist() for k, v in f.items()}

    dim_importances = {}
    for dim_idx, top_items in atlas.items():
        if not top_items: continue
        mean_score = sum(item["score"] for item in top_items) / len(top_items)
        dim_importances[dim_idx] = mean_score

    scores = dim_importances.values()

    plt.figure(figsize=(12, 6))
    xs = np.arange(len(scores))
    plt.bar(xs, scores, color='skyblue', label="Feature Importance")
    plt.xlabel('Features (sorted by importance)')
    plt.ylabel('Mean Activation')
    plt.title('Feature Importance Distribution')
    plt.legend()
    plt.savefig(f'{savedir}/sae_feature_importance.png', dpi=300)
    plt.close()
    print(f" Saved feature importance histogram to {savedir}/sae_feature_importance.png")


def get_all_activations(
        feature_extractor_name: Tuple[str, str],
        sae_model: nn.Module,
        dataset_name: str,
        device: str,
        batch_size: int = 64,
        aggregation: Literal['mean', 'binary_sum'] = 'binary_sum',
        threshold: float = 0.01,
        savedir: str = "."
) -> torch.Tensor:
    """
    Computes aggregated SAE activations for the entire dataset.
    This version correctly handles the ViT-style model pipeline.
    """
    feature_extractor, prep = load_encoder(feature_extractor_name[0], feature_extractor_name[1], device)
    feature_extractor.to(device).eval()
    sae_model.to(device).eval()

    dm = ImageDataManager(initialize=dataset_name, transform=prep)
    dataloader, _ = dm.get_dataloaders(dataset=dataset_name, batch_size=batch_size, shuffle=False)

    all_scores = []
    print("Step 1: Getting all dataset activations for clustering...")
    with torch.inference_mode():
        for images, _ in tqdm(dataloader, desc="Getting Activations"):
            images = images.to(device)

            if "clip" in str(type(feature_extractor)).lower():
                feature_extractor.visual.output_tokens = True
                _ , patch_embeddings = feature_extractor.visual.forward(images)
            elif "dino" in str(type(feature_extractor)).lower():
                x = feature_extractor.forward_features(images)
                patch_embeddings = x["x_norm_patchtokens"]
            else:
                patch_embeddings = feature_extractor(images)

            activations = sae_model.encode(patch_embeddings) # Shape: (B, Num_Patches, SAE_Dim)

            for i in range(images.shape[0]):
                item_activations = activations[i]
                scores = aggregate_activations(item_activations, aggregation, threshold)
                all_scores.append(scores)

    print(f"Saving activation scores to {savedir}/all_activations.pt")
    torch.save(torch.stack(all_scores, dim=0), f"{savedir}/all_activations.pt")
    return torch.stack(all_scores, dim=0)

def analyze_and_cluster_features(
        all_activations: torch.Tensor,
        atlas_path: str,
        k: int = 100,
        savedir: str = "plots",
        cluster_threshold: float = 0.6
) -> Tuple[np.ndarray, List[int]]:
    """
    Ranks features, correlates the top k, and saves cluster assignments.
    """
    print(f"\nStep 2: Analyzing top {k} feature correlations and clustering...")

    npz_file = np.load(atlas_path, allow_pickle=True)
    atlas = {int(key): val.tolist() for key, val in npz_file.items()}

    ranked_dims = rank_dimensions(atlas)
    top_k_indices = ranked_dims[:k]

    top_k_activations = all_activations[:, top_k_indices]

    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = np.corrcoef(top_k_activations.cpu().numpy(), rowvar=False)
    correlation_matrix = np.nan_to_num(correlation_matrix)

    distance_matrix = 1 - np.abs(correlation_matrix)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    cluster_labels = fcluster(linkage_matrix, t=cluster_threshold, criterion="distance")
    feature_clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        original_dim_idx = top_k_indices[i] # Map local index back to original
        if str(cluster_id) not in feature_clusters:
            feature_clusters[str(cluster_id)] = []
        feature_clusters[str(cluster_id)].append(original_dim_idx)

    cluster_path = f'{savedir}/sae_clusters.json'
    with open(cluster_path, 'w') as f:
        json.dump(feature_clusters, f, indent=2)
    print(f"✅ Saved top-{k} feature clusters to {cluster_path}")

    clustermap_labels = [str(i) for i in top_k_indices]
    cluster_grid = sns.clustermap(
        correlation_matrix,
        row_linkage=linkage_matrix, col_linkage=linkage_matrix,
        xticklabels=clustermap_labels, yticklabels=clustermap_labels,
        cmap="coolwarm", center=0, figsize=(12, 12)
    )
    plt.suptitle(f"Hierarchical Clustering of Top {k} Features", y=1.02)
    plt.savefig(f'{savedir}/hierarchical_feature_clustering_top_{k}.png', dpi=300)
    plt.close()

    return linkage_matrix, top_k_indices

def plot_dendrogram_with_images(
        linkage_matrix: np.ndarray,
        top_k_indices: List[int],
        atlas_path: str,
        k_images: int = 3,
        savedir: str = "plots"
):
    """
    Plots the clustering dendrogram with top activating images for the top-k features.
    """
    print(f"\nStep 3: Generating dendrogram for top {len(top_k_indices)} features...")
    npz_file = np.load(atlas_path, allow_pickle=True)
    atlas = {int(k): v.tolist() for k, v in npz_file.items()}

    dendro = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro['leaves']

    fig = plt.figure(layout="constrained", figsize=(len(leaf_order) * 1.5, 10))
    gs = plt.GridSpec(2, len(leaf_order), figure=fig, height_ratios=[0.5, 2])

    ax_dendro = fig.add_subplot(gs[0, :])
    dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        labels=top_k_indices,
        leaf_rotation=90,
        leaf_font_size=8
    )
    ax_dendro.set_title("Feature Clustering Dendrogram", fontsize=16)

    for col_idx, local_leaf_idx in enumerate(leaf_order):
        original_feature_idx = top_k_indices[local_leaf_idx] # Map back to original index
        ax_img_grid = fig.add_subplot(gs[1, col_idx])
        top_items = atlas.get(original_feature_idx, [])[:k_images]

        if not top_items:
            ax_img_grid.text(0.5, 0.5, f'N: {original_feature_idx}\n(No images)', ha='center')
            ax_img_grid.axis("off")
            continue

        images_for_grid = []
        for item in top_items:
            try:
                img = Image.open(item['image_path']).convert("RGB").resize((64, 64))
                images_for_grid.append(torchvision.transforms.ToTensor()(img))
            except Exception:
                continue

        if not images_for_grid: continue

        grid_tensor = torchvision.utils.make_grid(images_for_grid, nrow=1, padding=2)
        ax_img_grid.imshow(grid_tensor.permute(1, 2, 0).cpu().numpy())
        ax_img_grid.set_title(f'N: {original_feature_idx}', fontsize=10)
        ax_img_grid.axis("off")

    path = f'{savedir}/dendrogram_with_images.png'
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved dendrogram with images to {path}")


def get_umap_decoder_embeddings(sae, savedir):
    decoder_weights = sae.W_dec.detach().cpu().numpy()
    print(f"Extracted decoder weights with shape: {decoder_weights.shape}")
    print("Running UMAP algorithm... (this takes a while)")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine')
    embeddings_2d = reducer.fit_transform(decoder_weights)
    print(f"Computed 2D embeddings with shape: {embeddings_2d.shape}")
    np.save(f"{savedir}/umap", embeddings_2d)
    print(f"✅ UMAP embeddings saved successfully to: {savedir}/umap.np")