import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
import umap
from PIL import Image
from sae.core import SAE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm


COLORMAP = "coolwarm"

@dataclass
class StatsConfig:
    backbone_path: str = "../ckpts/clip/openai-clip-vit-base-patch16-cub-best_model"
    sae_path: str = "../ckpts/sae/sae_topk_d12288"
    dataset: str | list = "cub" #field(default_factory=lambda: ["cub"])
    data_root: str = "../../../data/"
    device: str = "cuda"
    k: int = 10
    save_directory = "../results/sae_stats"
    layer_index: int = -1
    cls_only: bool = False
    aggregation_method: str = "mean"
    num_workers: int = 12
    atlas_threshold: float = 0.1
    k_clusters: int = 100
    cluster_threshold: float = 0.6

def initialize_storage_tensors(
    d_sae: int, k: int
) -> Dict[str, torch.Tensor]:
    """Initialize tensors for storing results on the CPU."""
    return {
        "max_activating_image_values": torch.zeros([d_sae, k]),
        "max_activating_image_indices": torch.zeros([d_sae, k], dtype=torch.long),
        "sparsity": torch.zeros([d_sae]),
        "mean_acts": torch.zeros([d_sae]),
    }

def aggregate_embeddings(
    sae_latents: torch.Tensor, 
    method: str = 'binarize',
    binarize_threshold: float = 0.1 # Threshold for the binarize method
) -> torch.Tensor:
    """
    Aggregates patch/token-level SAE latents into image-level latents.

    Args:
        sae_latents: Tensor of shape [batch_size, num_tokens, sae_dim]
        method: 'max', 'mean', or 'binarize'.
                'max': Max activation across all patches. Good for "does this exist?"
                'mean': Mean activation. Good for "how prevalent is this?"
                'binarize': Counts how many patches activated a feature
                                   above the given binarize_threshold.
        binarize_threshold: The threshold to use for the 'binarize' method.
    Returns:
        Tensor of shape [batch_size, sae_dim]
    """
    if method == 'max':
        return torch.max(sae_latents, dim=1).values
    elif method == 'mean':
        return torch.mean(sae_latents, dim=1)
    elif method == 'binarize':
        binarized_latents = (sae_latents > binarize_threshold).float()
        return torch.sum(binarized_latents, dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def compute_sae_embedding(sae, backbone_embeddings, aggregation_method: str = "mean"):
    embeddings = sae.encode(backbone_embeddings)
    if aggregation_method == 'none':
        return embeddings
    if len(embeddings.size()) > 2:
        embeddings = aggregate_embeddings(embeddings, method=aggregation_method)
    return embeddings

# TODO: this is coupled to the backbone having a processor -> change in future
def compute_backbone_embeddings(
        backbone, 
        image_batch, 
        device, 
        layer_index: int = -1, 
        cls_only: bool = False
    ):
    inputs = backbone.processor(
        images=image_batch,
        return_tensors="pt",
        padding=True
    ).to(device)

    embeddings, _ = backbone(**inputs, return_patch_embeddings=True, layer_index=layer_index)

    if cls_only:
        embeddings = embeddings[:, 0, :]
    return embeddings

def compute_sae_statistics(sae_acts):
    mean_acts = sae_acts.sum(dim=1)
    sparsity = (sae_acts > 0).sum(dim=1)
    return mean_acts, sparsity

def compute_topk_activations(sae_acts, k: int):
    top_k = min(k, sae_acts.size(1))
    values, indices = torch.topk(sae_acts, k=top_k, dim=1)
    return values, indices

def get_new_top_k(
    first_values: torch.Tensor,
    first_indices: torch.Tensor,
    second_values: torch.Tensor,
    second_indices: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top k values and indices from two sets of values/indices."""
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices

# TODO: this is dependent on the compute_backbone_embeddings() function which is coupled to a certain backbone type
def process_batch(
        batch,
        backbone,
        sae,
        cfg,
        images_processed,
        storage,
        backbone_cache # Pass in the cache
):
    backbone_embedding = compute_backbone_embeddings(
        backbone=backbone, 
        image_batch=batch[0], 
        device=cfg.device, 
        cls_only=cfg.cls_only, 
        layer_index=cfg.layer_index
    )

    if not cfg.cls_only:
        # We have patch data (shape [B, T, D_model])
        for i in range(len(batch[0])):
            global_index = images_processed + i
            backbone_cache[global_index] = backbone_embedding[i, 1:, :].cpu()

    sae_embedding = compute_sae_embedding(
        sae=sae,
        backbone_embeddings=backbone_embedding, 
        aggregation_method=cfg.aggregation_method
    ).transpose(0,1) # Shape: [d_sae, batch_size]
    
    mean_acts, sparsity = compute_sae_statistics(sae_embedding)
    
    storage["mean_acts"] += mean_acts.detach().cpu()
    storage["sparsity"] += sparsity.detach().cpu()

    vals_gpu, indices_gpu_local = compute_topk_activations(
        sae_acts=sae_embedding,
        k=cfg.k,
    )
    indices_gpu_global = (indices_gpu_local + images_processed).long()
    
    old_vals_gpu = storage["max_activating_image_values"].to(cfg.device)
    old_indices_gpu = storage["max_activating_image_indices"].to(cfg.device)

    new_top_values_gpu, new_top_indices_gpu = get_new_top_k(
        old_vals_gpu,
        old_indices_gpu,
        vals_gpu,
        indices_gpu_global,
        cfg.k,
    )

    storage["max_activating_image_values"] = new_top_values_gpu.detach().cpu()
    storage["max_activating_image_indices"] = new_top_indices_gpu.detach().cpu()

    return storage, backbone_cache, sae_embedding

def save_stats(
        storage,
        cfg
): 
    if isinstance(cfg.dataset, list):
        dataset_name = "-".join(cfg.dataset)
    else:
        dataset_name = cfg.dataset
    full_save_dir = os.path.join(cfg.save_directory, dataset_name, cfg.sae_path.split("/")[-1])
    os.makedirs(full_save_dir, exist_ok=True)

    torch.save(
        storage["max_activating_image_indices"],
        f"{full_save_dir}/max_activating_image_indices.pt",
    )
    torch.save(
        storage["max_activating_image_values"],
        f"{full_save_dir}/max_activating_image_values.pt",
    )
    torch.save(storage["sparsity"], f"{full_save_dir}/sparsity.pt")
    torch.save(storage["mean_acts"], f"{full_save_dir}/mean_acts.pt")    

def analyze_and_cluster_features(
        top_k_activations: torch.Tensor,
        top_k_indices: List[int],
        savedir: str,
        metric_method: tuple,
        cluster_threshold: float = 0.6
) -> np.ndarray:
    """
    Correlates top k features using the specified metric and saves clusters.
    
    Args:
        top_k_activations: The continuous activation slice, shape [N_images, k_features]
        top_k_indices: The original indices of the k features (list of length k)
        savedir: Path to save plots and cluster files.
        metric: Similarity metric to use, either 'pearson' or 'jaccard'.
        cluster_threshold: The distance threshold for forming clusters.
    """
    k = len(top_k_indices)
    metric, method = metric_method
    
    if metric == 'pearson':
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity_matrix = np.corrcoef(top_k_activations.cpu().numpy(), rowvar=False)
        similarity_matrix = np.nan_to_num(similarity_matrix)
        cbar_label = 'Pearson Correlation'
    
    elif metric == 'jaccard':
        B = (top_k_activations > 0).cpu().numpy().astype(float)
        
        intersection_matrix = B.T @ B
        activations_per_feature = B.sum(axis=0)
        sum_matrix = activations_per_feature[:, np.newaxis] + activations_per_feature[np.newaxis, :]
        union_matrix = sum_matrix - intersection_matrix
        
        similarity_matrix = intersection_matrix / (union_matrix + 1e-9)
        cbar_label = 'Jaccard Similarity'
    
    else:
        raise ValueError("metric must be 'pearson' or 'jaccard'")

    if metric == 'pearson':
        distance_matrix = 1 - np.abs(similarity_matrix)
    else: # Jaccard
        distance_matrix = 1 - similarity_matrix
        
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    linkage_matrix = linkage(squareform(distance_matrix), method="ward")

    cluster_labels = fcluster(linkage_matrix, t=cluster_threshold, criterion="distance")
    
    feature_clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        original_dim_idx = top_k_indices[i] # Map local index back to original
        if str(cluster_id) not in feature_clusters:
            feature_clusters[str(cluster_id)] = []
        feature_clusters[str(cluster_id)].append(int(original_dim_idx))

    cluster_path = f'{savedir}/sae_clusters_top{k}_{metric}_{method}.json'
    with open(cluster_path, 'w') as f:
        json.dump(feature_clusters, f, indent=2)

    clustermap_labels = [str(i) for i in top_k_indices]
    cluster_grid = sns.clustermap(
        similarity_matrix,
        row_linkage=linkage_matrix, col_linkage=linkage_matrix,
        xticklabels=clustermap_labels, yticklabels=clustermap_labels,
        cmap=COLORMAP, center=0, figsize=(12, 12),
        cbar_kws={'label': cbar_label}
    )
    plt.suptitle(f"Hierarchical Clustering of Top {k} Features ({metric.title()})", y=1.02)
    plt.savefig(f'{savedir}/hierarchical_clustering_top_{k}_{metric}_{method}.png', dpi=300)
    plt.close()

    return linkage_matrix

def plot_dendrogram_with_images(
        linkage_matrix: np.ndarray,
        top_k_indices: List[int],
        atlas_path: str,
        k_images: int = 3,
        savedir: str = "plots",
        metric: tuple = ("jaccard", "common")
):
    """
    Plots the clustering dendrogram with top activating images for the top-k features.
    """
    try:
        with open(atlas_path, 'r') as f:
            atlas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not load atlas.json from {atlas_path}")
        return

    dendro = dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro['leaves']

    fig = plt.figure(layout="constrained", figsize=(len(leaf_order) * 1.5, 10))
    gs = plt.GridSpec(2, len(leaf_order), figure=fig, height_ratios=[0.5, 2])

    ax_dendro = fig.add_subplot(gs[0, :])
    dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        labels=[str(i) for i in top_k_indices],
        leaf_rotation=90,
        leaf_font_size=8
    )
    ax_dendro.set_title("Feature Clustering Dendrogram", fontsize=16)

    for col_idx, local_leaf_idx in enumerate(leaf_order):
        original_feature_idx = top_k_indices[local_leaf_idx] # Map back to original index
        ax_img_grid = fig.add_subplot(gs[1, col_idx])
        
        feature_key = f"feature_{original_feature_idx}"
        top_items = atlas.get(feature_key, [])[:k_images]

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

    path = f'{savedir}/dendrogram_with_images_{metric[0]}_{metric[1]}.png'
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def rank_dimensions(stats_path: str, sort_by: str = "top_mean_acts") -> list[int]:
    """
    Loads saved stats and returns a sorted list of all feature indices.
    """
    try:
        sparsity = torch.load(f"{stats_path}/sparsity.pt").cpu()
        mean_acts = torch.load(f"{stats_path}/mean_acts.pt").cpu()
        max_vals = torch.load(f"{stats_path}/max_activating_image_values.pt")
    except FileNotFoundError:
        print(f"Error: Could not load stats files from {stats_path}")
        return []

    all_indices = torch.arange(len(sparsity))


    if sort_by == "top_k_strength":
        scores = torch.mean(max_vals, dim=1)
        sort_descending = True
    elif sort_by == "top_mean_acts":
        scores = mean_acts # Total sum
        sort_descending = True
    else: # Default to sorting by index
        scores = all_indices
        sort_descending = False

    sorted_score_indices = torch.argsort(scores, descending=sort_descending)
    final_sorted_feature_indices = all_indices[sorted_score_indices]
    
    return final_sorted_feature_indices.numpy().tolist()

@torch.inference_mode()
def run_stats(cfg, dataset, backbone):
    dataset_name = "-".join(cfg.dataset) if isinstance(cfg.dataset, list) else cfg.dataset
    stats_path = os.path.join(cfg.save_directory, dataset_name, cfg.sae_path.split("/")[-1])

    sae = SAE.load(cfg.sae_path).to(cfg.device)

    def _collate(batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return images, labels
    
    dl = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=_collate)

    storage = initialize_storage_tensors(d_sae=sae.cfg.d_sae, k=cfg.k)
    backbone_cache = {}
    all_sae_activations_list = []
    num_processed = 0

    for batch in tqdm(dl, desc="Fetching Activations"):
        storage, backbone_cache, sae_embedding = process_batch(
            batch=batch,
            backbone=backbone,
            sae=sae,
            cfg=cfg,
            images_processed=num_processed,
            storage=storage,
            backbone_cache=backbone_cache
        )
        all_sae_activations_list.append(sae_embedding.transpose(0, 1).cpu())
        num_processed += len(batch[0])

    storage["mean_acts"] /= (storage["sparsity"] + 1e-9)
    storage["sparsity"] /= num_processed

    save_stats(storage, cfg)

    if not backbone_cache:
        print("Skipping atlas generation: Stats were run with cls_only=True.")
    else:
        generate_atlas(
            cfg=cfg,
            sae=sae,
            dataset=dataset,
            backbone_cache=backbone_cache
        )

    
    all_activations = torch.cat(all_sae_activations_list, dim=0)
    del all_sae_activations_list
    ranked_dims_common = rank_dimensions(stats_path, sort_by="top_mean_acts")
    top_k_indices_common = ranked_dims_common[:cfg.k_clusters]
    top_k_common_slice = all_activations[:, top_k_indices_common]
    
    torch.save(
        top_k_common_slice, 
        f"{stats_path}/top_{cfg.k_clusters}_common_activations.pt"
    )
    torch.save(
        top_k_indices_common, 
        f"{stats_path}/top_{cfg.k_clusters}_common_indices.pt"
    )

    ranked_dims_strong = rank_dimensions(stats_path, sort_by="top_k_strength")
    top_k_indices_strong = ranked_dims_strong[:cfg.k_clusters]
    top_k_strong_slice = all_activations[:, top_k_indices_strong]
    
    torch.save(
        top_k_strong_slice, 
        f"{stats_path}/top_{cfg.k_clusters}_strong_activations.pt"
    )
    torch.save(
        top_k_indices_strong, 
        f"{stats_path}/top_{cfg.k_clusters}_strong_indices.pt"
    )
    
    del all_activations, top_k_common_slice, top_k_strong_slice

    run_all_clustering_from_checkpoint(cfg)
    
    return storage, backbone_cache

@torch.inference_mode()
def generate_atlas(cfg, sae, dataset, backbone_cache):
    d_sae = sae.cfg.d_sae
    dataset_name = "-".join(cfg.dataset) if isinstance(cfg.dataset, list) else cfg.dataset

    stats_path = os.path.join(cfg.save_directory, dataset_name, cfg.sae_path.split("/")[-1])
    storage = {"indices": torch.load(f"{stats_path}/max_activating_image_indices.pt")}

    atlas_data = {}
    
    for feature_idx in tqdm(range(d_sae), desc="Building JSON"):
        feature_key = f"feature_{feature_idx}"
        atlas_data[feature_key] = []
        
        topk_img_indices = storage["indices"][feature_idx].long().cpu().tolist()

        embeddings_to_process = [backbone_cache[idx] for idx in topk_img_indices]
        
        backbone_batch_gpu = torch.stack(embeddings_to_process).to(cfg.device)
        
        sae_embedding = compute_sae_embedding(
            sae=sae, 
            backbone_embeddings=backbone_batch_gpu,
            aggregation_method='none' # Get patch-level data
        ) # Shape: [k, T_patches, d_sae]

        patch_data_batch = sae_embedding[:, :, feature_idx].cpu() # Shape: [k, T_patches]

        for i, global_index in enumerate(topk_img_indices):
            if cfg.dataset == "cub": # TODO: ew
                img_name = dataset.data.iloc[global_index].image_name
                image_path = os.path.join(cfg.data_root, "CUB_200_2011/images", img_name)
            else:
                image_path = dataset.df.iloc[global_index][dataset.labelkey]
                        
            patch_activations_tensor = patch_data_batch[i]
            
            active_indices = torch.where(patch_activations_tensor > cfg.atlas_threshold)[0]
            
            active_indices_list = active_indices.numpy().tolist()
            
            active_values = patch_activations_tensor[active_indices].numpy().tolist()

            atlas_data[feature_key].append({
                "image_path": image_path,
                "active_patch_indices": active_indices_list,
                "active_values": active_values
            })

    save_dir = os.path.join(cfg.save_directory, dataset_name, cfg.sae_path.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    
    save_file = os.path.join(save_dir, f"atlas.json")
    
    with open(save_file, 'w') as f:
        json.dump(atlas_data, f)

def run_all_clustering_from_checkpoint(cfg):
    """
    Runs all 4 clustering analyses (2 rankings x 2 metrics)
    from the saved checkpoints.
    """
    dataset_name = "-".join(cfg.dataset) if isinstance(cfg.dataset, list) else cfg.dataset
    stats_path = os.path.join(cfg.save_directory, dataset_name, cfg.sae_path.split("/")[-1])
    atlas_path = os.path.join(stats_path, f"atlas.json")

    rank_methods = {
        "common": (
            f"{stats_path}/top_{cfg.k_clusters}_common_activations.pt",
            f"{stats_path}/top_{cfg.k_clusters}_common_indices.pt"
        ),
        "strong": (
            f"{stats_path}/top_{cfg.k_clusters}_strong_activations.pt",
            f"{stats_path}/top_{cfg.k_clusters}_strong_indices.pt"
        )
    }
    
    metrics = ['pearson', 'jaccard']

    for rank_name, (acts_path, indices_path) in rank_methods.items():
        try:
            top_k_activations = torch.load(acts_path)
            top_k_indices = torch.load(indices_path)
        except FileNotFoundError:
            print(f"Error: Could not find checkpoint files for k={cfg.k_clusters}, rank='{rank_name}'.")
            print("Please run the full 'run_stats_and_generate_atlas' function first.")
            continue
            
        for metric in metrics:
            
            linkage_matrix = analyze_and_cluster_features(
                top_k_activations=top_k_activations,
                top_k_indices=top_k_indices,
                savedir=stats_path,
                metric_method=(metric, rank_name),
                cluster_threshold=cfg.cluster_threshold
            )

            plot_dendrogram_with_images(
                linkage_matrix=linkage_matrix,
                top_k_indices=top_k_indices,
                atlas_path=atlas_path, 
                k_images=3,
                savedir=stats_path,
                metric=(metric, rank_name)
            )
    
def plot_firing_frequencies(sparsity):
    """Plots firing frequency histogram of living features."""
    total_features = sparsity.numel()
    dead_features_mask = (sparsity == 0)
    num_dead = dead_features_mask.sum().item()
    percent_dead = 100 * num_dead / total_features

    print(f"Total Features: {total_features}")
    print(f"Dead Features:  {num_dead} ({percent_dead:.2f}%)")

    living_sparsities = sparsity[~dead_features_mask]

    if living_sparsities.numel() > 0:
        log_frequencies = torch.log10(living_sparsities).cpu().numpy()

        plt.hist(log_frequencies, bins=100)
        plt.xlabel("Log10 Activation Frequency")
        plt.ylabel("# Features")
        plt.show()
    else:
        print("All features are dead.")

def plot_firing_frequency_against_mean_activation(sparsity, mean_acts):
    """Plots firing frequency against the mean activation value for each living feature."""
    dead_features_mask = (sparsity == 0)
    num_dead = dead_features_mask.sum().item()
    print(f"Number of dead features: {num_dead}")

    living_sparsities = sparsity[~dead_features_mask]
    living_mean_acts = mean_acts[~dead_features_mask]

    if living_sparsities.numel() > 0:
        log_freq = torch.log10(living_sparsities + 1e-9).cpu().numpy()
        log_mean_act = torch.log10(living_mean_acts + 1e-9).cpu().numpy()

        plt.scatter(log_freq, log_mean_act, alpha=0.1, s=5)
        plt.xlabel("Log10 Activation Frequency")
        plt.ylabel("Log10 Mean Activation Value")
        plt.show()
    else:
        print("All features are dead, nothing to plot.")

@torch.inference_mode()
def generate_and_save_umap(sae_path: str, save_dir: str):
    """
    Loads an SAE, runs UMAP on its decoder weights, and saves the
    2D embeddings as 'umap_embeddings.pt'.
    """
    print("Starting UMAP generation...")
    
    sae_model = SAE.load(sae_path).to("cpu")
    decoder_weights = sae_model.W_dec.detach().cpu().numpy()
    
    num_features = decoder_weights.shape[0]
    print(f"Running UMAP on {num_features} decoder vectors...")

    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='cosine',
        verbose=True
    )
    embeddings_2d = reducer.fit_transform(decoder_weights)
    
    save_path = os.path.join(save_dir, "umap_embeddings.pt")
    torch.save(torch.tensor(embeddings_2d), save_path)
    
    print(f"UMAP embeddings saved to {save_path}")

def visualize_atlas(
    atlas_path: str,
    feature_dims: list[int],
    mode: str = "clear",
    k_to_show: int = 5,
    img_size: int = 224,
    patch_size: int = 16
):
    """
    Visualizes the top-k activating images for given SAE features.

    Args:
        atlas_path: Path to the generated 'atlas.json' file.
        feature_dims: A list of feature indices to visualize (e.g., [10, 42, 1234]).
        mode: Visualization mode. One of ['whole', 'heatmap', 'mask'].
        k_to_show: How many of the top-k images to display (default: 5).
        img_size: The resolution the images were processed at (default: 224).
        patch_size: The patch size of the Vision Transformer (default: 16).
    """
    
    if mode not in ['clear', 'heatmap', 'mask']:
        raise ValueError("Mode must be one of 'whole', 'heatmap', or 'mask'")

    try:
        with open(atlas_path, 'r') as f:
            atlas = json.load(f)
    except FileNotFoundError:
        print(f"Error: Atlas file not found at {atlas_path}")
        return

    num_features = len(feature_dims)
    if num_features == 0:
        print("No feature dimensions provided.")
        return

    fig, axs = plt.subplots(
        num_features, 
        k_to_show, 
        figsize=(k_to_show * 3, num_features * 3.2),
        squeeze=False
    )
    
    num_patches_per_dim = img_size // patch_size
    num_patches_total = num_patches_per_dim * num_patches_per_dim

    for i, feature_idx in enumerate(feature_dims):
        feature_key = f"feature_{feature_idx}"
        if feature_key not in atlas:
            print(f"Warning: Feature {feature_idx} not in atlas. Skipping.")
            for j in range(k_to_show): axs[i, j].axis("off")
            continue
        
        top_k_entries = atlas[feature_key]
        
        for j in range(k_to_show):
            ax = axs[i, j]
            ax.axis("off")
            
            if j >= len(top_k_entries):
                continue
                
            entry = top_k_entries[j]
            img_path = entry["image_path"]

            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
            except FileNotFoundError:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=8, color='red')
                continue

            if i == 0:
                ax.set_title(f"Image {j+1}", fontsize=10)
            if j == 0:
                ax.text(-0.1, 0.5, f"Feat {feature_idx}", transform=ax.transAxes, ha='right', va='center', fontsize=12, rotation=90)

            if mode == 'clear':
                ax.imshow(img)
                continue

            indices = entry.get("active_patch_indices", [])
            values = entry.get("active_values", [])
            
            if not indices:
                ax.imshow(img)
                continue

            dense_activations = np.zeros(num_patches_total)
            dense_activations[indices] = values
            heatmap_2d = dense_activations.reshape((num_patches_per_dim, num_patches_per_dim))

            if mode == 'heatmap':
                heatmap_img = Image.fromarray(heatmap_2d).resize((img_size, img_size), Image.NEAREST)
                
                ax.imshow(img, alpha=1.0)
                heatmap_norm = (heatmap_img - np.min(heatmap_img)) / (np.max(heatmap_img) - np.min(heatmap_img) + 1e-9)
                ax.imshow(heatmap_norm, cmap='hot', alpha=0.5)

            elif mode == 'mask':
                mask_2d = (heatmap_2d > 0).astype(np.uint8)
                mask_img = Image.fromarray(mask_2d * 255).resize((img_size, img_size), Image.NEAREST)
                
                img_np = np.array(img)
                mask_np = np.array(mask_img) / 255.0
                masked_img = img_np * np.expand_dims(mask_np, axis=-1)
                
                ax.imshow(masked_img.astype(np.uint8))

    fig.suptitle(f"Atlas Visualization (Mode: '{mode.upper()}')", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.show()