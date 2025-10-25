from src.eval.linear_probe import evaluate_linear_probe
from src.eval.knn import evaluate_knn
from src.eval.subgroups import evaluate_subgroups
from src.utils import load_config

def run_fm_evaluation(config_path: str):
    """Runs the evaluation suite for trained foundation models."""
    cfg = load_config(config_path)
    print("Starting evaluation...")

    evaluate_linear_probe(cfg)
    #evaluate_knn(config)
    #evaluate_subgroups(config)

    print("Evaluation completed.")

def run_sae_evaluation(config: dict):
    """Runs the evaluation suite for trained SAE models."""


    print("Starting SAE evaluation...")

    # sae eval funcs

    print("SAE Evaluation completed.")