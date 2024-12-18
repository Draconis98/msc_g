"""Main module for running model evaluation sweeps with configurable parameters."""

import logging
from argument_parser import parse_args
from sweep_runner import run_sweep
from utils.misc import setup_logging, setup_environment

def main():
    """Main entry point of the application."""
    # Setup logging
    setup_logging()
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup environment (e.g., mirror configuration)
        setup_environment(args.seed, args.use_mirror)

        # Run the sweep
        run_sweep(args)
        
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        raise

if __name__ == "__main__":
    main()