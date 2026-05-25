import yaml
from typing import Any, Dict
from pathlib import Path


def load_config(filename: str="hyperparameter.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns it as a Python dictionary.

    Args:
        filename (str): Name of the YAML configuration file.

    Returns:
        dict: Parsed configuration parameters.
    """
    
    # Path to the directory containing THIS file: src/config/
    config_dir = Path(__file__).parent

    # Create full path to the YAML file
    config_path = config_dir / filename

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    return config
