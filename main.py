"""Main module for running model training and evaluation sweeps with configurable parameters.

This module is the entry point of the application.
It parses the command line arguments, sets up the environment, and runs the sweep.

Usage:
    python main.py -s fft,lora -m Qwen/Qwen3-0.6B,Qwen/Qwen2-0.5B -d cais/mmlu -lr 1e-4,1e-5 -ed mmlu_gen,gsm8k_gen -e 1,2 -tt CAUSAL_LM -r 16,32 -t q_proj,k_proj,v_proj
"""

import os
from loguru import logger

from parse import parse_args
from sweep import run

def setup_environment(use_mirror=True):
    """Setup environment variables and configurations.
    
    Args:
        use_mirror (bool): Whether to use mirror for downloading models.
    """

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Use mirror for downloading models")
    else:
        logger.warning("Use default endpoint for downloading models")



def main():
    """Main entry point of the application."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment (e.g., mirror configuration)
    setup_environment(args.use_mirror)

    # Run the sweep
    run(args)

if __name__ == "__main__":
    main()