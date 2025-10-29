import yaml

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns: 
        dict: Configuration parameters as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_configs(base_config: dict, sweep_config: dict) -> dict:
    """
    Merges the sweep configuration into the base configuration.
    Values from sweep_config will override values in base_config.
    """
    merged_config = base_config.copy()
    merged_config.update(sweep_config)
    return merged_config