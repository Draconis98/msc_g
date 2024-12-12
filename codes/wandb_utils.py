import wandb
import os
import yaml
from datetime import datetime
from config import WANDB_CONFIG, SWEEP_LOGS_DIR

def setup_wandb_run(config):
    """Initialize a new wandb run with given configuration."""
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        config=config,
        name=f"{config['model_name']}-{config['strategy']}-{config['dataset']}"
    )
    return run

def create_sweep_config(args):
    """Create a wandb sweep configuration from command line arguments."""
    sweep_config = {
        'entity': WANDB_CONFIG["entity"],
        'project': WANDB_CONFIG["project"],
        'program': '/home/draco/graduation/codes/pipeline.py',
        'method': 'grid',
        'parameters': {
            key: {'values': [getattr(args, key)] if not isinstance(getattr(args, key), list) else getattr(args, key)}
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
        'values': [args.target_modules] if isinstance(args.target_modules, list) else [[args.target_modules]]
    }
    
    return sweep_config

def save_sweep_config(sweep_config):
    """Save sweep configuration to a file and return the filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_filename = f"sweep_{timestamp}.yaml"
    
    os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
    filepath = os.path.join(SWEEP_LOGS_DIR, sweep_filename)
    
    with open(filepath, 'w') as file:
        yaml.dump(sweep_config, file, default_flow_style=False)