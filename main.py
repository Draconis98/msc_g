"""Main module for running model training and evaluation sweeps with configurable parameters.

This module is the entry point of the application.
It parses the command line arguments, sets up the environment, and runs the sweep.

Usage:
    python main.py -s fft,lora -m Qwen/Qwen3-0.6B,Qwen/Qwen2-0.5B -d cais/mmlu -lr 1e-4,1e-5 -ed mmlu_gen,gsm8k_gen -e 1,2 -tt CAUSAL_LM -r 16,32 -t q_proj,k_proj,v_proj
"""

import os
import random
import numpy as np
import torch
from loguru import logger

from parse import parse_args
from sweep import run

def setup_environment(seed=42, use_mirror=True):
    """Setup environment variables and configurations.
    
    Args:
        seed (int): Random seed for reproducibility.
        use_mirror (bool): Whether to use mirror for downloading models.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set seed to {seed}")

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Use mirror for downloading models")
    else:
        logger.warning("Use default endpoint for downloading models")

def main():
    """Main entry point of the application."""
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup environment (e.g., mirror configuration)
        setup_environment(args.seed, args.use_mirror)

        # Run the sweep
        run(args)
        
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise

if __name__ == "__main__":
    main()