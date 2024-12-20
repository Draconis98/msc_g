"""Main module for running model evaluation sweeps with configurable parameters."""

import logging
from argument_parser import parse_args
from sweep_runner import run_sweep
from utils.misc import set_seed, setup_environment

def main():
    """Main entry point of the application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Parse command line arguments
        logging.info("Parsing command line arguments...")
        args = parse_args()
        
        # Setup environment (e.g., mirror configuration)
        logging.info("Setting up environment...")
        setup_environment(args.use_mirror)
        
        # Setup seed
        set_seed(args.seed)
        
        # Run the sweep
        logging.info("Starting sweep...")
        run_sweep(args)
        
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        raise

if __name__ == "__main__":
    main()