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
parser.add_argument(
    '--sweep', 
    type=str, 
    default="assets/configs/sweep.yaml",
    help="Path to the sweep definition YAML file."
    )
parser.add_argument(
    "--do_sweep",
    action="store_true",
    help="Whether to run a wandb sweep."
)
parser.add_argument(
    "--sweep_count",
    type=int,
    default=50,
    help="How many sweep runs to run."
)

ARGS = parser.parse_args()


def main():
    if ARGS.mode == "train":
        from src.train import run_training
        run_training(ARGS)
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
