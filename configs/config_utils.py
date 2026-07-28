# =============================================================================
# Configuration Utilities
# =============================================================================
# This module provides a helper function to load and merge YAML configuration
# files. It combines a base configuration (base.yaml) with a model‑specific
# YAML (e.g., timesformer.yaml) so that model‑specific overrides take precedence.
# =============================================================================

import yaml
import os


def load_config(config_path: str, base_path: str = None) -> dict:
    """
    Load and merge a model-specific YAML configuration with a base configuration.

    The base configuration (base.yaml) contains shared hyperparameters (training,
    data, logging). The model-specific YAML can override any key from the base.
    Nested dictionaries are merged recursively.

    Args:
        config_path (str): Path to the model-specific YAML file
                           (e.g., "configs/timesformer.yaml").
        base_path (str, optional): Path to the base YAML file. If not provided,
                                   defaults to "base.yaml" in the same directory
                                   as this script.

    Returns:
        dict: Merged configuration dictionary where values from the model-specific
              file override those in the base.
    """
    # If no base path is given, assume base.yaml resides in the same folder
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "base.yaml")

    # Load base configuration
    with open(base_path, 'r') as f:
        base = yaml.safe_load(f)

    # Load model‑specific configuration
    with open(config_path, 'r') as f:
        model_cfg = yaml.safe_load(f)

    # Recursively merge the model‑specific configuration into the base.
    # For each top‑level key in the model config:
    #   - If the value is a dictionary and the same key exists in the base
    #     as a dictionary, update the nested dictionary recursively.
    #   - Otherwise (non‑dict or key missing in base), replace the base value.
    for key, value in model_cfg.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key].update(value)          # Merge nested dicts
        else:
            base[key] = value                # Overwrite or add new key

    return base