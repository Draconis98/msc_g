"""Module for managing and executing W&B sweeps across multiple GPUs."""

import os
import time
import logging
import wandb
from utils.config import SWEEP_LOGS_DIR
from utils.gpu import is_gpu_free
from utils.wandb import create_sweep_config, save_sweep_config
from utils.misc import setup_logging

setup_logging()

class SweepRunner:
    """Manages and executes W&B sweeps across multiple GPUs."""
    
    def __init__(self, args):
        """Initialize SweepRunner with command line arguments."""
        self.args = args
        self.sweep_id = None
        self.api = None
        self.sweep_config = None
        self.expected_run_count = None
        self.api = wandb.Api()

    def _setup_configuration(self):
        """Setup W&B sweep configuration."""
        try:
            self.sweep_config = create_sweep_config(self.args)
            save_sweep_config(self.sweep_config)
            
            # Create sweep using wandb API
            self.sweep_id = wandb.sweep(self.sweep_config, project=self.sweep_config['project'])
        except Exception as e:
            logging.error("Failed to setup sweep: %s", str(e))
            raise

    def start_agent(self, gpu_id):
        """Start a W&B agent on specified GPU."""
        try:
            runtime_log_filename = os.path.join(
                SWEEP_LOGS_DIR,
                f'sweep_{self.sweep_id}_gpu{gpu_id}_runtime.log'
            )
            
            # Ensure logs directory exists
            os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
            
            # Run agent using sweep_id
            os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} nohup wandb agent {self.sweep_id} > {runtime_log_filename} 2>&1 &")
            logging.info("Started agent on GPU%d", gpu_id)
        except Exception as e:
            logging.error("Failed to start agent on GPU%d: %s", gpu_id, str(e))
            raise

    def _get_expected_run_count(self):
        """Get the expected run count of the sweep."""
        try:
            sweep_path = f"{self.sweep_config['project']}/{self.sweep_id}"
            sweep = self.api.sweep(sweep_path)

            self.expected_run_count = sweep.expected_run_count
            return self.expected_run_count
        
        except wandb.errors.CommError as e:
            logging.error("Network error while accessing sweep: %s", str(e))
            return False
        except KeyError as e:
            logging.error("Invalid sweep config: missing key %s", str(e))
            return False
        except AttributeError as e:
            logging.error("Sweep object missing expected attribute: %s", str(e))
            return False

    def run(self):
        """Run the sweep on available GPUs."""
        self._setup_configuration()

        try:
            logging.info(self._get_expected_run_count())

            for gpu_id in range(2):  # Assuming we want to use first 2 GPUs
                if self.expected_run_count == 0:
                    break

                if is_gpu_free(gpu_id):
                    self.start_agent(gpu_id)
                    self.expected_run_count -= 1
                else:
                    logging.warning("GPU%d is busy, skipping...", gpu_id)
        except Exception as e:
            logging.error("Error running sweep: %s", str(e))
            raise

def run_sweep(args):
    """Main function to setup and run a wandb sweep."""
    runner = SweepRunner(args)
    runner.run()