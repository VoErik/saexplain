# G-AUDIT Implementation
# Reference: https://arxiv.org/abs/2302.11382
# Note: This implementation does not include the upper bounding method for the risk score.

from typing import Union

import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
from PIL import Image
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm


class AttributeDataset(Dataset):
    """A simple dataset to return an image and a single attribute label."""
    def __init__(self, df, attribute_key, transform=None):
        self.df = df.dropna(subset=[attribute_key]).reset_index(drop=True)
        self.transform = transform
        self.attribute_key = attribute_key
        self.labels = self.df[self.attribute_key].astype('category').cat
        self.class_map = {i: cat for i, cat in enumerate(self.labels.categories)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels.codes[idx]
        return img, label
    
def bootstrap_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap_samples: int = 1000, confidence_level: float = 0.95):
    """
    Calculates bootstrap confidence intervals for accuracy.
    """
    rng = np.random.default_rng(42)
    accuracies = []
    
    for _ in range(n_bootstrap_samples):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        accuracies.append(accuracy_score(sample_true, sample_pred))
        
    mean_acc = np.mean(accuracies)
    lower_bound = np.percentile(accuracies, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(accuracies, (1 + confidence_level) / 2 * 100)
    
    return mean_acc, lower_bound, upper_bound

def bootstrap_nmi(series1: np.ndarray, series2: np.ndarray, n_bootstrap_samples: int = 1000, confidence_level: float = 0.95):
    """Calculates bootstrap confidence intervals for NMI."""
    rng = np.random.default_rng(42)
    nmis = []
    
    arr1 = series1
    arr2 = series2
    
    for _ in range(n_bootstrap_samples):
        indices = rng.choice(len(arr1), len(arr1), replace=True)
        sample1 = arr1[indices]
        sample2 = arr2[indices]
        nmis.append(normalized_mutual_info_score(sample1, sample2))
        
    mean_nmi = np.mean(nmis)
    lower_bound = np.percentile(nmis, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(nmis, (1 + confidence_level) / 2 * 100)
    
    return mean_nmi, lower_bound, upper_bound

def calculate_utility(df: pd.DataFrame, target_label: str, attributes_to_test: list) -> dict:
    """Calculates the statistical utility (NMI) with bootstrap CIs."""
    utility_results = {}
    for attr in attributes_to_test:
        subset_df = df.dropna(subset=[attr, target_label])
        
        attr_array = subset_df[attr].to_numpy()
        target_array = subset_df[target_label].to_numpy()
        
        mean_nmi, lower_ci, upper_ci = bootstrap_nmi(attr_array, target_array)
        
        utility_results[attr] = (mean_nmi, lower_ci, upper_ci)
        
    return utility_results

def _train_one_fold(dataset, train_indices, test_indices, num_classes, epochs):
    """
    Trains and evaluates a probe model on a single fold of data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (1 - step / num_training_steps) ** 0.7)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_preds, all_labels

def calculate_predictions_for_detectability(
    df: pd.DataFrame, 
    target_label: str, 
    attribute: str, 
    condition_on_target: bool,
    n_splits: int = 5, 
    epochs: int = 1
):
    """
    Helper function that returns the raw predictions (Â) and true labels (A) for a single attribute.
    """
    aggregated_preds = []
    aggregated_true_attributes = []
    
    if condition_on_target:
        # --- CONDITIONAL PATH ---
        for label_class in df[target_label].unique():
            partition_df = df[df[target_label] == label_class]
            attr_dataset = AttributeDataset(partition_df, attribute_key=attribute, transform=None)
            if len(attr_dataset.class_map) <= 1: continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_partition = attr_dataset.labels.codes
            for train_indices, test_indices in skf.split(np.zeros(len(y_partition)), y_partition):
                preds, labels = _train_one_fold(attr_dataset, train_indices, test_indices, len(attr_dataset.class_map), epochs)
                aggregated_preds.extend(preds)
                original_values = attr_dataset.labels.categories[labels].to_numpy()
                aggregated_true_attributes.extend(original_values)
    else:
        # --- UNCONDITIONAL PATH ---
        attr_dataset = AttributeDataset(df, attribute_key=attribute, transform=None)
        if len(attr_dataset.class_map) <= 1: return np.array([]), np.array([])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y = attr_dataset.labels.codes
        for train_indices, test_indices in skf.split(attr_dataset, y):
            preds, labels = _train_one_fold(attr_dataset, train_indices, test_indices, len(attr_dataset.class_map), epochs)
            aggregated_preds.extend(preds)
            original_values = attr_dataset.labels.categories[labels].to_numpy()
            aggregated_true_attributes.extend(original_values)

    # Convert integer predictions back to original attribute values
    full_attr_map = {i: cat for i, cat in enumerate(df[attribute].astype('category').cat.categories)}
    preds_as_values_np = np.array([full_attr_map.get(p) for p in aggregated_preds])
    
    return np.array(aggregated_true_attributes), preds_as_values_np

def run_g_audit(df: pd.DataFrame, target_label: str, attributes_to_test: list, is_anti_causal: bool) -> pd.DataFrame:
    """
    Runs the full, paper-compliant G-AUDIT analysis.
    For anti-causal tasks, final detectability is MI(A, Â_cond).
    """
    print("--- Running G-AUDIT: Calculating Utility (with Bootstrap) ---")
    utility_results = calculate_utility(df, target_label, attributes_to_test)
    
    detectability_results = {}
    for attr in attributes_to_test:
        print(f"\n--- Calculating Detectability for '{attr}' ---")
        
        if is_anti_causal:
            # --- ANTI-CAUSAL PATH (Y -> X) ---
            print("  Scenario: Anti-Causal (Y -> X). Using conditional model for final metric.")
            
            # 1. Get conditional predictions (the "fair" model)
            print("  Running conditional model to get Â_cond...")
            true_attributes, cond_preds = calculate_predictions_for_detectability(
                df, target_label, attr, condition_on_target=True, epochs=10
            )
            
            # The final detectability metric is MI(A, Â_cond)
            if len(cond_preds) == 0:
                detectability_results[attr] = (0.0, 0.0, 0.0)
                continue

            mean_nmi, lower_ci, upper_ci = bootstrap_nmi(true_attributes, cond_preds)
            detectability_results[attr] = (mean_nmi, lower_ci, upper_ci)
            print(f"  Final Detectability MI(A, Â_cond): {mean_nmi:.4f} (95% CI: [{lower_ci:.4f}, {upper_ci:.4f}])")

        else:
            # --- CAUSAL PATH (X -> Y) ---
            print("  Scenario: Causal (X -> Y). Using unconditional model for final metric.")
            
            # 1. Get unconditional predictions
            true_attributes, uncond_preds = calculate_predictions_for_detectability(
                df, target_label, attr, condition_on_target=False, epochs=10
            )

            # The final detectability metric is MI(A, Â_uncond)
            if len(uncond_preds) == 0:
                detectability_results[attr] = (0.0, 0.0, 0.0)
                continue

            mean_nmi, lower_ci, upper_ci = bootstrap_nmi(true_attributes, uncond_preds)
            detectability_results[attr] = (mean_nmi, lower_ci, upper_ci)
            print(f"  Final Detectability MI(A, Â_uncond): {mean_nmi:.4f} (95% CI: [{lower_ci:.4f}, {upper_ci:.4f}])")

    # --- Assemble the final DataFrame ---
    data = []
    for attr in attributes_to_test:
        mean_nmi_u, u_lower, u_upper = utility_results.get(attr, (0.0, 0.0, 0.0))
        mean_nmi_d, d_lower, d_upper = detectability_results.get(attr, (0.0, 0.0, 0.0))
        data.append({
            'Attribute': attr,
            'Utility (Mean NMI)': mean_nmi_u, 'Utility_CI_Lower': u_lower, 'Utility_CI_Upper': u_upper,
            'Detectability (Mean NMI)': mean_nmi_d, 'Detectability_CI_Lower': d_lower, 'Detectability_CI_Upper': d_upper
        })
    
    risk_df = pd.DataFrame(data)
    risk_df['Risk Score'] = risk_df['Utility (Mean NMI)'] * risk_df['Detectability (Mean NMI)']
    risk_df = risk_df.sort_values('Risk Score', ascending=False)
    
    return risk_df


def plot_utility_detectability(risk_df: pd.DataFrame, save_path: Union[str, Path] = "./g-audit.png"):
    """
    Plots Utility (NMI) vs. Detectability (NMI) with 95% confidence intervals.

    This function is designed to visualize the results from the rigorous,
    paper-compliant G-AUDIT implementation.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 9))

    risk_df['x_err_lower'] = risk_df['Utility (Mean NMI)'] - risk_df['Utility_CI_Lower']
    risk_df['x_err_upper'] = risk_df['Utility_CI_Upper'] - risk_df['Utility (Mean NMI)']
    risk_df['y_err_lower'] = risk_df['Detectability (Mean NMI)'] - risk_df['Detectability_CI_Lower']
    risk_df['y_err_upper'] = risk_df['Detectability_CI_Upper'] - risk_df['Detectability (Mean NMI)']

    attributes = sorted(risk_df['Attribute'].unique())
    palette = sns.color_palette(n_colors=len(attributes))
    color_map = dict(zip(attributes, palette))

    sns.scatterplot(
        x='Utility (Mean NMI)',
        y='Detectability (Mean NMI)',
        hue='Attribute',
        data=risk_df,
        legend='full',
        alpha=0.8,
        marker='D',
        palette=color_map,
        s=150
    )
    
    for _, row in risk_df.iterrows():
        plt.errorbar(
            x=row['Utility (Mean NMI)'],
            y=row['Detectability (Mean NMI)'],
            xerr=[[row['x_err_lower']], [row['x_err_upper']]],
            yerr=[[row['y_err_lower']], [row['y_err_upper']]], 
            fmt='s',
            color=color_map[row['Attribute']],
            capsize=5, 
            elinewidth=1.5, 
            markeredgewidth=1.5, 
            alpha=0.6,
            zorder=-1
        )
        plt.text(
            row['Utility (Mean NMI)'],
            row['Detectability (Mean NMI)'] + 0.02,
            row['Attribute'], 
            fontsize=11, 
            ha='center', 
            va='bottom'
        )

    plt.title('G-AUDIT: Attribute Detectability vs. Utility (with 95% CI)', fontsize=16)
    plt.xlabel('Utility', fontsize=12)
    plt.ylabel('Detectability', fontsize=12)

    plt.xlim(0, risk_df['Utility_CI_Upper'].max() + 0.1)
    plt.ylim(0, risk_df['Detectability_CI_Upper'].max() + 0.1)

    plt.legend(title='Attribute', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()