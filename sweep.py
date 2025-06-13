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
        self.debug = args.debug

    def _create_sweep_config(self):
        """Create a wandb sweep configuration from command line arguments.
        
        Returns:
            dict: A dictionary of sweep configuration.
        """
        try:
            config = {
                'entity': self.args.wandb_entity,
                'project': self.args.wandb_project,
                'method': self.args.sweep_method,
                'parameters': {
                    key: {'values': [getattr(self.args, key)] 
                        if not isinstance(getattr(self.args, key), list) 
                        else getattr(self.args, key)}
                    for key in [
                        # Training strategy and model configuration
                        'strategy', 'model_name', 'dataset', 'eval_dataset',
                        # Batch processing and saving configuration
                        'save_steps', 'save_total_limit',
                        # Gradient and optimization configuration
                        'gradient_checkpointing', 'gradient_accumulation_steps', 'warmup_ratio', 
                        # Data processing configuration
                        'packing', 'max_seq_length', 'padding_free', 'max_out_len',
                        # Output and precision configuration
                        'overwrite_output_dir', 'bf16', 'use_cache', 'attn_implementation',
                        # Task and debugging configuration
                        'task_type', 'dataset_batched', 'seed', 'enable_thinking', 
                    ]
                }
            }

            if config['method'] != 'grid':
                config.update({
                    'metric': {
                        'name': self.args.sweep_metric,
                        'goal': 'maximize' if self.args.sweep_metric in \
                            ['eval_accuracy', 'eval_f1', 'eval_rouge', 'eval_bleu', 'eval_meteor', 'eval_bertscore', \
                             'eval_rouge_l', 'eval_rouge_l_summary', 'eval_rouge_l_summary_f1', 'eval_rouge_l_summary_recall', \
                             'eval_rouge_l_summary_precision'] else 'minimize'
                    }
                })

                min_lr = min(self.args.learning_rate)
                max_lr = max(self.args.learning_rate)
                config['parameters']['learning_rate'] = {'min': min_lr, 'max': max_lr}
                for param in ['batch_size', 'epochs', 'rank']:
                    param_value = getattr(self.args, param)
                    config['parameters'][param] = {
                        'values': [param_value] if isinstance(param_value, list) else [[param_value]]
                    }
            else:
                for param in ['learning_rate', 'batch_size', 'epochs', 'rank']:
                    config['parameters'][param] = {
                        'values': getattr(self.args, param)
                    }

            # Handle target_modules separately
            if self.args.strategy != 'fft':
                config['parameters']['target_modules'] = {
                    'values': [self.args.target_modules] if isinstance(self.args.target_modules, list) else [[self.args.target_modules]]
                }
            
            config['parameters']['eval_dataset'] = {
                'values': [self.args.eval_dataset] if isinstance(self.args.eval_dataset, list) else [[self.args.eval_dataset]]
            }
            
            self.sweep_config = config
        except ValueError as e:
            raise ValueError(f"Invalid sweep configuration: {e}") from e
        except Exception as e:
            logger.error("Failed to create sweep configuration: %s", str(e))
            raise
    
    def _setup_configuration(self):
        """Setup W&B sweep configuration.
        
        Returns:
            str: The sweep ID.
        """
        try:
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
            wandb.agent(self.sweep_id, function=pipeline(debug=self.debug), project=self.sweep_config['project'])
        except Exception as e:
            logger.error("Failed to start agent: %s", str(e))
            raise

    def run(self):
        """Run the sweep on available GPUs."""
        self._create_sweep_config()
        self._setup_configuration()
        if self.args.sweep_method == 'grid':
            logger.info(f"Expected run count: {self._get_expected_run_count()}")

        self._start_agent()


class ResumeRun:
    """Class to handle resuming a specific W&B run."""
    
    def __init__(self, run_id, eval_dataset, debug):
        """Initialize the ResumeRun with command line arguments.
        
        Args:
            args (argparse.Namespace): Command line arguments containing run_id.
        """
        self.api = wandb.Api()
        self.run_id = run_id
        self.eval_dataset = eval_dataset
        self.debug = debug
        
    def _get_run_info(self):
        """Get run information and validate its status.
        
        Returns:
            wandb.Run: W&B run object or None if cannot resume
        """
        try:
            wandb_run = self.api.run(self.run_id)
            
            logger.info(f"Found run: {wandb_run.name}")

            if self.eval_dataset is not None:
                logger.info(f"Update eval_dataset: {self.eval_dataset}")
                wandb_run.config['eval_dataset'] = self.eval_dataset
                wandb_run.update()

            return wandb_run
            
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
        pipeline(resume=True, run_id=self.run_id, debug=self.debug)
        logger.success("Run resumption completed")

# WANDB: Weights and Biases
def run(args):
    """Main function to setup and run a wandb sweep.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    if args.resume:
        ResumeRun(args.run_id, args.eval_dataset, args.debug).run()
    else:
        Sweep(args).run()