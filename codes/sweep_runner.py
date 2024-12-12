import os
import time
import logging
import wandb
from config import SWEEP_LOGS_DIR
from gpu_utils import is_gpu_free
from wandb_utils import create_sweep_config, save_sweep_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SweepRunner:
    def __init__(self, args):
        """Initialize SweepRunner with command line arguments."""
        self.args = args
        self.sweep_id = None
        self.api = None
        self.sweep_config = None
        
        self.api = wandb.Api()

    def setup_sweep(self):
        """Setup W&B sweep configuration."""
        try:
            self.sweep_config = create_sweep_config(self.args)
            save_sweep_config(self.sweep_config)
            
            # Create sweep using wandb API
            self.sweep_id = wandb.sweep(
                self.sweep_config, 
                project=self.sweep_config['project']
            )
            
            # Wait for sweep to be created
            logging.info("Waiting for sweep to be initialized...")
            time.sleep(10)
        except Exception as e:
            logging.error(f"Failed to setup sweep: {str(e)}")
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
            logging.info(f"Started agent on GPU{gpu_id}")
            
            # Give the agent some time to start
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"Failed to start agent on GPU{gpu_id}: {str(e)}")
            raise

    def check_sweep_status(self):
        """Check the status of the sweep."""
        try:
            sweep = self.api.sweep(f"{self.sweep_config['entity']}/{self.sweep_config['project']}/{self.sweep_id}")
            pending_runs = [run for run in sweep.runs if run.state == "pending"]
            return len(pending_runs) > 0
        except Exception as e:
            logging.error(f"Error checking sweep status: {e}")
            return False

    def run(self):
        """Run the sweep on available GPUs."""
        try:
            self.setup_sweep()
            
            # Run agents on available GPUs
            agents_started = False
            for gpu_id in range(2):  # Assuming we want to use first 2 GPUs
                if is_gpu_free(gpu_id):
                    self.start_agent(gpu_id)
                    agents_started = True
                    
                    if not self.check_sweep_status():
                        break
                else:
                    logging.warning(f"GPU{gpu_id} is busy, skipping...")
            
            if not agents_started:
                raise RuntimeError("No available GPUs found to run agents")
            
            logging.info("Sweep completed. No more pending runs.")
            
        except Exception as e:
            logging.error(f"Error running sweep: {str(e)}")
            raise

def setup_environment(use_mirror=True):
    """Setup environment variables and configurations."""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Ensure required directories exist
    os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)

def run_sweep(args):
    """Main function to setup and run a wandb sweep."""
    runner = SweepRunner(args)
    runner.run() 