"""Module for managing and executing W&B sweeps across multiple GPUs."""

import os
import logging
from datetime import datetime
import yaml
import wandb

from utils.misc import setup_logging
from utils.config import WANDB_CONFIG, SWEEP_LOGS_DIR

from pipeline import pipeline

setup_logging()

class SweepRunner:
    """Manages and executes W&B sweeps across multiple GPUs."""
    
    def __init__(self, args):
        """Initialize SweepRunner with command line arguments."""
        self.args = args
        self.sweep_id = None
        self.sweep_config = None
        self.sweep_path = None
        self.expected_run_count = None
        self.api = wandb.Api()

    def _create_sweep_config(self):
        """Create a wandb sweep configuration from command line arguments."""
        try:
            sweep_config = {
                'entity': WANDB_CONFIG["entity"],
                'project': WANDB_CONFIG["project"],
            'method': 'grid',
            'parameters': {
                key: {'values': [getattr(self.args, key)] if not isinstance(getattr(self.args, key), list) else getattr(self.args, key)}
                for key in [
                    'strategy', 'model_name', 'task', 'dataset', 'eval_dataset',
                    'learning_rate', 'learning_schedule', 'rank', 'epochs',
                    'batch_size', 'save_steps', 'save_total_limit',
                    'gradient_checkpointing', 'gradient_accumulation_steps',
                    'warmup_ratio', 'packing', 'max_seq_length',
                    'overwrite_output_dir', 'bf16', 'use_cache',
                    'dataset_batched', 'seed'
                ]
            }
        }
        
            # Handle target_modules separately
            sweep_config['parameters']['target_modules'] = {
                'values': [self.args.target_modules] if isinstance(self.args.target_modules, list) else [[self.args.target_modules]]
            }
            
            self.sweep_config = sweep_config
        except Exception as e:
            logging.error("Failed to create sweep configuration: %s", str(e))
            raise
    
    def _save_sweep_config(self):
        """Save sweep configuration to a file and return the filename."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sweep_filename = f"sweep_{timestamp}.yaml"
            
            os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
            filepath = os.path.join(SWEEP_LOGS_DIR, sweep_filename)
            
            with open(filepath, 'w', encoding='utf-8') as file:
                yaml.dump(self.sweep_config, file, default_flow_style=False) 
        except Exception as e:
            logging.error("Failed to save sweep configuration: %s", str(e))
            raise

    def _setup_configuration(self):
        """Setup W&B sweep configuration."""
        self._create_sweep_config()
        self._save_sweep_config()

        try:
            # Create sweep using wandb API
            self.sweep_id = wandb.sweep(self.sweep_config, project=self.sweep_config['project'])
        except Exception as e:
            logging.error("Failed to setup sweep: %s", str(e))
            raise

    def _get_expected_run_count(self):
        """Get the expected run count of the sweep."""
        try:
            self.sweep_path = f"{self.sweep_config['entity']}/{self.sweep_config['project']}/{self.sweep_id}"
            self.expected_run_count = self.api.sweep(self.sweep_path).expected_run_count
            return self.expected_run_count
        
        except wandb.errors.CommError as e:
            logging.error("Network error while accessing sweep: %s", str(e))
            raise
        except KeyError as e:
            logging.error("Invalid sweep config: missing key %s", str(e))
            raise
        except AttributeError as e:
            logging.error("Sweep object missing expected attribute: %s", str(e))
            raise
    
    def _start_agent(self):
        """Start a W&B agent on specified GPUs."""
        try:
            wandb.agent(self.sweep_id, function=pipeline, project=self.sweep_config['project'])
        except Exception as e:
            logging.error("Failed to start agent: %s", str(e))
            raise

    def run(self):
        """Run the sweep on available GPUs."""
        self._setup_configuration()
        logging.info("Expected run count: %d", self._get_expected_run_count())

        self._start_agent()

def run_sweep(args):
    """Main function to setup and run a wandb sweep."""
    runner = SweepRunner(args)
    runner.run()