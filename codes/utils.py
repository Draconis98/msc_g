import os
import random
import logging
import numpy as np
import torch

def setup_logging(log_file=None, format='%(asctime)s - %(levelname)s - %(message)s'):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=handlers
    )

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def create_run_name(config):
    """Create a standardized run name from configuration."""
    return "{}-{}-{}-{}-{}-r{}-{}epochs-seed{}".format(
        config['strategy'], config['model_name'], config['task'], 
        config['dataset'], config['learning_rate'], config['rank'], 
        config['epochs'], config['seed']
    )

def ensure_dir(directory):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory 