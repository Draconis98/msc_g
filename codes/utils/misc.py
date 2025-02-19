import os
import random
import numpy as np
import torch
from loguru import logger
from utils.config import OUTPUT_DIR

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set seed to {seed}")

def setup_environment(seed, use_mirror=True):
    """Setup environment variables and configurations."""
    set_seed(seed)

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Use mirror for downloading models")
    else:
        logger.warning("Use default endpoint for downloading models")


def ensure_dir(directory):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensure directory {directory} exists")
    return directory 

def get_output_dir(config):
    """Get the output directory for the training."""
    if config['data_selection']:
        return os.path.join(
            OUTPUT_DIR,
            f"{config['strategy']}/{config['task']}/{config['dataset']}/" \
            f"{config['model_name'].replace(':', '/')}/" \
            f"{config['learning_rate']}/r{config['rank']}/{config['epochs']}epochs-ds"
        )
    else:
        return os.path.join(
            OUTPUT_DIR,
            f"{config['strategy']}/{config['task']}/{config['dataset']}/" \
            f"{config['model_name'].replace(':', '/')}/" \
            f"{config['learning_rate']}/r{config['rank']}/{config['epochs']}epochs"
        )