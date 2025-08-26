from typing import Dict, List, Optional, Union, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

######################################
### INTERPRETABILITY FUNCTIONALITY ###
######################################
# TODO: make this actually work and maybe put in eval?

@torch.no_grad()
def get_all_feature_activations(
        inference_model: nn.Module, # Expects the combined InferenceModel
        data_loader: DataLoader
) -> List[Dict[str, Union[int, torch.Tensor]]]:
    """
    Stage 1: Collects raw feature activations using the integrated InferenceModel.
    """
    inference_model.eval()

    all_activations = []

    for batch in tqdm(data_loader, total=len(data_loader), desc="Collecting Activations"):
        image_ids_batch, images_batch, _ = batch
        images = images_batch.to(next(inference_model.parameters()).device)

        # Use the model's method to get SAE features for the whole batch
        # Assumes extract_sae_features returns shape [B, num_patches, d_sae]
        sae_features_batch = inference_model.extract_sae_features(images)

        for j in range(sae_features_batch.shape[0]):
            image_id = image_ids_batch[j].item()
            feature_acts = sae_features_batch[j]

            all_activations.append({
                "image_id": image_id,
                "feature_acts": feature_acts.cpu()
            })

    return all_activations

@torch.no_grad()
def analyze_image_activations(
        inference_model: nn.Module, # Expects the combined InferenceModel
        image: torch.Tensor,
        top_k_features: int = 5,
        top_k_patches: int = 5
) -> Dict[int, List[tuple[int, float]]]:
    """
    For a single image, finds its most active features and their top patches.
    """
    inference_model.eval()

    # Get SAE features for the single image. Shape: [1, num_patches, d_sae]
    feature_acts = inference_model.extract_sae_features(image.unsqueeze(0)).squeeze(0)

    max_activations_per_feature, _ = torch.max(feature_acts, dim=0)
    top_feature_indices = torch.topk(max_activations_per_feature, k=top_k_features).indices

    results = {}
    for feature_idx in top_feature_indices:
        acts_for_feature = feature_acts[:, feature_idx.item()]
        top_patch_values, top_patch_indices = torch.topk(acts_for_feature, k=top_k_patches)
        results[feature_idx.item()] = list(zip(top_patch_indices.tolist(), top_patch_values.tolist()))

    return results

def aggregate_and_rank_images(all_activations, feature_idx, aggregation_method, threshold):
    image_scores = {}
    for record in all_activations:
        image_id = record['image_id']
        acts_for_feature = record['feature_acts'][:, feature_idx]
        if aggregation_method == 'count': score = (acts_for_feature > threshold).sum().item()
        elif aggregation_method == 'mean': score = acts_for_feature.mean().item()
        else: raise ValueError()
        image_scores[image_id] = score
    return sorted(image_scores.items(), key=lambda item: item[1], reverse=True)


def get_top_images_for_features(
        all_activations: List[Dict],
        top_k_images: int = 5,
        feature_indices: Optional[List[int]] = None,
        num_top_features_to_analyze: int = 0,
        aggregation_method: str = 'count',
        activation_threshold: float = 0.01,
) -> Dict[int, List[tuple[int, float]]]:
    """
    Finds the top activating images for a given set of SAE features by calling
    the 'aggregate_and_rank_images' helper function.
    """
    if feature_indices is None and num_top_features_to_analyze > 0:
        # Automatically find the most frequently activating features
        print(f"Finding the {num_top_features_to_analyze} most active features...")
        total_activations_per_feature = torch.zeros(all_activations[0]['feature_acts'].shape[1])
        for record in all_activations:
            total_activations_per_feature += record['feature_acts'].sum(dim=0)
        feature_indices = torch.topk(total_activations_per_feature, k=num_top_features_to_analyze).indices.tolist()
        print(f"Analyzing features: {feature_indices}")

    elif feature_indices is None:
        raise ValueError("You must provide either feature_indices or num_top_features_to_analyze.")

    results = {}
    for feature_idx in tqdm(feature_indices, desc="Ranking images per feature"):
        top_images = aggregate_and_rank_images(
            all_activations,
            feature_idx=feature_idx,
            aggregation_method=aggregation_method,
            threshold=activation_threshold
        )
        # Store the top K results
        results[feature_idx] = top_images[:top_k_images]

    return results