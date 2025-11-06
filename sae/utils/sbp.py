import math
import torch
import torch.nn.functional as F

def calculate_sbp_loss(activation_matrix, num_classes, 
                       method:str = "entropy", eps: float = 1e-8, 
                       alpha: float = 0.5, support_threshold: float = 0):
    if method == "entropy":
        loss = calculate_entropy_support_loss(
            activation_matrix=activation_matrix
            )
    elif method == "uniform":
        loss = calculate_uniformity_loss(
            activation_matrix=activation_matrix, num_classes=num_classes, eps=eps
            )
    elif method == "power_normalized":
        loss = calculate_power_target_loss(
            activation_matrix=activation_matrix, alpha=alpha
            )
    elif method == "alpha_mixing":
        loss = calculate_alpha_target_loss(
            activation_matrix=activation_matrix, alpha=alpha, support_threshold=support_threshold
            )
    else:
        raise NotImplementedError(f"{method} not implemented.")
    return loss

def calculate_uniformity_loss(activation_matrix, num_classes: int, eps: float = 1e-8):
    """
    Encourages each feature to activate uniformly over all classes.

    Args:
        activation_matrix (Tensor): A (num_features x num_classes) matrix,
            where entry (i, c) is the total activation of feature i for class c
            in the current (or EMA-smoothed) batch.
        num_classes (int): Number of classes in the dataset.
        eps (float): Numerical stability constant.

    Returns:
        Tensor: Scalar tensor representing mean KL divergence to the uniform distribution.

    Math:
        For each feature j, compute its empirical distribution:
            p_{c,j} = activation[c] / sum_c activation[c]

        Let u_c = 1 / C be the uniform distribution over classes.
        The KL divergence is:
            KL(p_j || u) = sum_c p_{c,j} log(p_{c,j} / u_c)
                         = sum_c p_{c,j} log p_{c,j} + log C

        Minimizing KL pushes p_j closer to uniform over **all** classes.
    """
    log_p = F.log_softmax(activation_matrix, dim=1)
    P = torch.exp(log_p)
    kl_to_uniform = torch.sum(P * log_p, dim=1) + math.log(num_classes + eps)
    return torch.mean(kl_to_uniform)


def calculate_alpha_target_loss(activation_matrix, alpha: float = 0.5, support_threshold: float = 0.0):
    """
    Encourages each feature to activate uniformly over its empirical support only.

    Args:
        activation_matrix (Tensor): (num_features x num_classes) activation counts.
        gamma (float): Mixing coefficient between empirical distribution and uniform-over-support.
                       - gamma = 0   → no regularization
                       - gamma = 1   → perfectly uniform over support

    Returns:
        Tensor: Scalar mean KL divergence from empirical p to target q.

    Math:
        Let p_j be the empirical distribution of feature j:
            p_{c,j} = activation[c] / sum_c activation[c]

        Support S_j = {c | p_{c,j} > 0}
        Uniform over support:
            u_{c,j} = 1 / |S_j|   if c ∈ S_j
                      0           otherwise

        Construct target distribution:
            q_{c,j} = (1 - γ) p_{c,j} + γ u_{c,j}

        Loss:
            KL(p_j || q_j) = sum_c p_{c,j} log(p_{c,j} / q_{c,j})

        Advantage:
            - Does NOT force uniform over all classes
            - Uniformity only inside the feature's own support
    """
    P = activation_matrix / (activation_matrix.sum(dim=1, keepdim=True) + 1e-10)

    support = (P > support_threshold).float()
    support_size = support.sum(dim=1, keepdim=True)

    U = support / (support_size + 1e-10)
    Q = (1 - alpha) * P + alpha * U

    kl = torch.sum(P * torch.log((P + 1e-10) / (Q + 1e-10)), dim=1)
    return torch.mean(kl)



def calculate_power_target_loss(activation_matrix, alpha: float = 0.5):
    """
    Forces each feature toward a power-normalized distribution of its empirical activations.

    Args:
        activation_matrix (Tensor): (num_features x num_classes)
        alpha (float): Power exponent. Smaller alpha moves p toward uniform.
                       - alpha = 1  → no regularization
                       - alpha → 0 → uniform over support

    Returns:
        Tensor: Scalar mean KL divergence KL(p || q).

    Math:
        Empirical: p_{c,j} = activation[c] / sum_c activation[c]

        Target:
            q_{c,j} = p_{c,j}^α / sum_{c'} p_{c',j}^α

        If p is very peaked, lowering α flattens q toward uniform.
        If p is already flat, q stays close to p.

        Loss:
            KL(p_j || q_j) = sum_c p_{c,j} log(p_{c,j} / q_{c,j})

        Effect:
            - Softly penalizes peaked class dependencies
            - Keeps feature support unchanged
    """
    P = activation_matrix / (activation_matrix.sum(dim=1, keepdim=True) + 1e-10)

    Q = P ** alpha
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-10)

    kl = torch.sum(P * torch.log((P + 1e-10) / (Q + 1e-10)), dim=1)
    return torch.mean(kl)



def calculate_entropy_support_loss(activation_matrix):
    """
    Maximizes the entropy of the empirical class distribution of each feature.

    Args:
        activation_matrix (Tensor): (num_features x num_classes)

    Returns:
        Tensor: Mean negative entropy. (Minimizing this maximizes entropy.)

    Math:
        p_{c,j} = activation / sum activation
        Entropy:
            H(p_j) = - sum_c p_{c,j} log(p_{c,j})
        We return -H(p) = sum_c p log p

        Minimizing this encourages p_j to be spread out,
        but ONLY over classes where it already has support.

        Difference from uniform/KL methods:
            - No explicit target distribution
            - Pure entropy maximization
            - If a feature activates on 3 species, it spreads across those 3,
              but never pushes it to activate where it was zero.
    """
    P = activation_matrix / (activation_matrix.sum(dim=1, keepdim=True) + 1e-10)

    log_p = torch.log(P + 1e-10)
    neg_entropy = torch.sum(P * log_p, dim=1)
    entropy = -neg_entropy
    return torch.mean(entropy)



