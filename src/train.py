import functools
import wandb
from src.utils import ExecTimer, load_config, merge_configs

def sweep_entrypoint(base_config_path: str):
    """
    This function is the entrypoint for a wandb agent.
    """
    from src.train_sae import train_sae
    with wandb.init() as run:
        sweep_config = run.config.as_dict()
        base_config = load_config(base_config_path)
        final_config = merge_configs(base_config, sweep_config)
        train_sae(final_config)
    

def run_training(args):
    """
    Run the training process based on the provided configuration file.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    
    print(f"Running training with config: {args.config}")

    cfg = load_config(args.config)

    if cfg.get("mode", None) in ["backbone", "backbone-classify"]:
        from src.train_backbone import train_backbone

        with ExecTimer("Backbone Training"):
            train_backbone(cfg)

    elif cfg.get("mode", None) == "sae":
        from src.train_sae import train_sae

        if args.do_sweep:
            print(f"Starting a wandb sweep.")
            sweep_cfg = load_config(args.sweep)
            sweep_id = wandb.sweep(sweep=sweep_cfg, project="thesis", entity="voerik")
            run_func = functools.partial(sweep_entrypoint, base_config_path=args.config)

            wandb.agent(sweep_id, function=run_func, count=args.sweep_count)
        else:
            with ExecTimer("SAE Training"):
                train_sae(cfg)

    elif cfg.get("mode", None) == "joint":
        from src.misc import train_joint
        with ExecTimer("Joint Training"):
            train_joint(cfg)
    
    else:
        raise ValueError(f"Unknown training mode: {cfg.get('mode', None)}")