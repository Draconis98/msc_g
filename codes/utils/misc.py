import os
import random
import logging
import numpy as np
import torch
from utils.config import OUTPUT_DIR, SWEEP_LOGS_DIR

def setup_logging(log_file=None, format='%(asctime)s - %(levelname)s - %(message)s'):
    """Setup logging configuration."""
    class ColorFormatter(logging.Formatter):
        """Custom formatter with colored output."""
        COLORS = {
            'INFO': '\033[92m',
            'WARNING': '\033[93m',
            'ERROR': '\033[91m',
            'RESET': '\033[0m'
        }

        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                record.levelname = colored_levelname
            return super().format(record)

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    for handler in handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColorFormatter(format))
        else:
            handler.setFormatter(logging.Formatter(format))

    logging.basicConfig(
        level=logging.INFO,
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

def setup_environment(use_mirror=True):
    """Setup environment variables and configurations."""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Ensure required directories exist
    os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
    
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

def get_output_dir(config):
    """Get the output directory for the training."""
    return os.path.join(
        OUTPUT_DIR,
        f"{config['strategy']}/{config['task']}/{config['dataset']}/" \
        f"{config['model_name'].replace(':', '/')}/" \
        f"{config['learning_rate']}/r{config['rank']}/{config['epochs']}epochs"
    )