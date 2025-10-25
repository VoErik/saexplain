import argparse

parser = argparse.ArgumentParser(
    description="Provide necessary config files for either running training, inference or interpretability research."
    )
parser.add_argument(
    "--mode", 
    type=str, 
    choices=["train", "inference", "interp", "eval"], 
    required=True, 
    help="Mode of operation: train, inference, or interp."
    )
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    help="Path to the config file."
    )

ARGS = parser.parse_args()

def main():
    if ARGS.mode == "train":
        from src.train import run_training
        run_training(ARGS.config)
    elif ARGS.mode == "inference":
        from src.inference import run_inference
        run_inference(ARGS.config)
    elif ARGS.mode == "interp":
        from src.interp_tools import run_interpretability
        run_interpretability(ARGS.config)
    elif ARGS.mode == "eval":
        from src.eval import run_fm_evaluation
        run_fm_evaluation(ARGS.config)
    else:
        raise ValueError(f"Unknown mode: {ARGS.mode}")


if __name__ == "__main__":
    main()
