"""Module for managing and executing W&B sweeps across multiple GPUs.

This module is responsible for creating a W&B sweep configuration,
setting up the sweep, and starting the agent.
"""

# https://docs.wandb.ai/guides/
from loguru import logger

import wandb
from pipeline import pipeline

class Sweep:
    """Manages and executes W&B sweeps across multiple GPUs."""
    
    def __init__(self, args):
        """Initialize Sweep with command line arguments.
        
        Args:
            args (argparse.Namespace): Command line arguments.
        
        Attributes:
            args (argparse.Namespace): Command line arguments.
            sweep_id (str): The ID of the sweep.
            sweep_config (dict): The configuration of the sweep.
            sweep_path (str): The path of the sweep.
            expected_run_count (int): The expected run count of the sweep.
        """
        self.args = args
        self.sweep_id = args.sweep_id
        self.sweep_config = None
        self.sweep_path = None
        self.expected_run_count = None
        self.api = wandb.Api()

    def _create_sweep_config(self):
        """Create a wandb sweep configuration from command line arguments.
        
        Returns:
            dict: A dictionary of sweep configuration.
        """
        try:
            config = {
                'entity': self.args.wandb_entity,
                'project': self.args.wandb_project,
                'method': 'grid',
                'parameters': {
                    key: {'values': [getattr(self.args, key)] 
                        if not isinstance(getattr(self.args, key), list) 
                        else getattr(self.args, key)}
                    for key in [
                        'strategy', 'model_name', 'dataset', 'eval_dataset',
                        'learning_rate', 'learning_schedule', 'rank', 'epochs',
                        'batch_size', 'save_steps', 'save_total_limit',
                        'gradient_checkpointing', 'gradient_accumulation_steps',
                        'warmup_ratio', 'packing', 'max_seq_length', 'padding_free',
                        'overwrite_output_dir', 'bf16', 'use_cache', 'attn_implementation',
                        'task_type', 'dataset_batched', 'seed', 'debug', 'enable_thinking', 
                    ]
                }
            }
            
            # Handle target_modules separately
            config['parameters']['target_modules'] = {
                'values': [self.args.target_modules] if isinstance(self.args.target_modules, list) else [[self.args.target_modules]]
            }
            
            config['parameters']['eval_dataset'] = {
                'values': [self.args.eval_dataset] if isinstance(self.args.eval_dataset, list) else [[self.args.eval_dataset]]
            }
            
            config['parameters']['learning_rate'] = {
                'values': self.args.learning_rate  # Use the list of learning rates
            }
            
            self.sweep_config = config
        except Exception as e:
            logger.error("Failed to create sweep configuration: %s", str(e))
            raise
    
    def _setup_configuration(self):
        """Setup W&B sweep configuration.
        
        Returns:
            str: The sweep ID.
        """
        try:
            if self.args.resume and self.args.sweep_id:
                logger.info(f"[Resume] Reuse the unfinished sweep: {self.args.sweep_id}")
                self.sweep_id = self.args.sweep_id
                self.sweep_config = self.api.sweep(self.sweep_config['entity'] + "/" + self.sweep_config['project'] + "/" + self.sweep_id).config
                return
            self.sweep_id = wandb.sweep(self.sweep_config, project=self.sweep_config['project'])
        except Exception as e:
            logger.error("Failed to setup sweep: %s", str(e))
            raise

    def _get_expected_run_count(self):
        """Get the expected run count of the sweep.
        
        Returns:
            int: The expected run count.
        """
        try:
            self.sweep_path = f"{self.sweep_config['entity']}/{self.sweep_config['project']}/{self.sweep_id}"
            self.expected_run_count = self.api.sweep(self.sweep_path).expected_run_count
            return self.expected_run_count
        
        except wandb.errors.CommError as e:
            logger.error("Network error while accessing sweep: %s", str(e))
            raise
        except KeyError as e:
            logger.error("Invalid sweep config: missing key %s", str(e))
            raise
        except AttributeError as e:
            logger.error("Sweep object missing expected attribute: %s", str(e))
            raise
    
    def _start_agent(self):
        """Start a W&B agent on specified GPUs."""
        try:
            wandb.agent(self.sweep_id, function=pipeline, project=self.sweep_config['project'])
        except Exception as e:
            logger.error("Failed to start agent: %s", str(e))
            raise

    def run(self):
        """Run the sweep on available GPUs."""
        self._create_sweep_config()
        self._setup_configuration()
        logger.info(f"Expected run count: {self._get_expected_run_count()}")

        self._start_agent()


class ResumeRun:
    """Class to handle resuming a specific W&B run."""
    
    def __init__(self, run_id, eval_dataset):
        """Initialize the ResumeRun with command line arguments.
        
        Args:
            args (argparse.Namespace): Command line arguments containing run_id.
        """
        self.api = wandb.Api()
        self.run_id = run_id
        self.eval_dataset = eval_dataset
        
    def _get_run_info(self):
        """Get run information and validate its status.
        
        Returns:
            wandb.Run: W&B run object or None if cannot resume
        """
        try:
            run = self.api.run(self.run_id)
            
            logger.info(f"Found run: {run.name}")

            if self.eval_dataset is not None:
                logger.info(f"Update eval_dataset: {self.eval_dataset}")
                run.config['eval_dataset'] = self.eval_dataset
                run.update()

            return run
            
        except wandb.errors.CommError as e:
            logger.error(f"Network error while accessing run: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get run information: {e}")
            raise
    
    def run(self):
        """Execute the run resumption process."""
        run_info = self._get_run_info()
        
        if run_info is None:
            logger.error("Cannot resume finished run")
            return
            
        logger.info(f"Starting to resume run: {self.run_id}")
        pipeline(resume=True, run_id=self.run_id)
        logger.success("Run resumption completed")

# WANDB: Weights and Biases
def run(args):
    """Main function to setup and run a wandb sweep.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    if args.resume:
        ResumeRun(args.run_id, args.eval_dataset).run()
    else:
        Sweep(args).run()