from datetime import datetime
import argparse
import os
import torch
import time
import logging
import wandb

from config import (
    ALLOWED_MODELS, ALLOWED_STRATEGIES, ALLOWED_TASKS, TRAINING_DEFAULTS, SWEEP_LOGS_DIR
)
from wandb_utils import (
    create_sweep_config, save_sweep_config
)

def parse_strategies(input_str):
    strategies = input_str.split(',')
    if not all(strategy in ALLOWED_STRATEGIES for strategy in strategies):
        raise argparse.ArgumentTypeError(f"Invalid strategy. Each strategy must be one of {ALLOWED_STRATEGIES}.")
    return strategies

def parse_model(input_str):
    models = input_str.split(',')
    for model in models:
        if ':' not in model:
            raise argparse.ArgumentTypeError(f"Invalid format. Each model must be in the format 'model_name:model_size'.")
        model_name, model_size = model.split(':')
        if model_name not in ALLOWED_MODELS or model_size not in ALLOWED_MODELS[model_name]:
            raise argparse.ArgumentTypeError(f"Invalid model or size. Allowed models and sizes are: {ALLOWED_MODELS}.")
    return models

def parse_args():
    parser = argparse.ArgumentParser(description="Process training parameters.")
    
    # Required arguments
    parser.add_argument('-s', '--strategy', type=parse_strategies, required=True,
                       help='Strategy to use (comma-separated list)')
    parser.add_argument('-m', '--model_name', type=parse_model, required=True,
                       help='Name of the model (format: model_name:size)')
    parser.add_argument('-t', '--task', type=str, required=True,
                       choices=ALLOWED_TASKS, help='Task to perform')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--eval_dataset', type=str, required=True,
                       help='Evaluation dataset')
    parser.add_argument('-lr', '--learning_rate', type=float, required=True,
                       help='Learning rate')
    parser.add_argument('-r', '--rank', type=lambda x: [int(i) for i in x.split(',')] if ',' in x else int(x),
                       required=True, help='Rank value (comma-separated list or single integer)')
    parser.add_argument('-e', '--epochs', type=lambda x: [int(i) for i in x.split(',')] if ',' in x else int(x),
                       required=True, help='Number of epochs (comma-separated list or single integer)')
    parser.add_argument('--target_modules', type=lambda x: x.split(','), required=True,
                       help='Target modules (comma-separated list)')
    
    # Optional arguments with defaults from config
    for key, value in TRAINING_DEFAULTS.items():
        parser.add_argument(f'--{key}', type=type(value), default=value,
                          help=f'{key.replace("_", " ").title()} (default: {value})')
    
    parser.add_argument('--use_mirror', type=bool, default=True,
                       help='Use mirror for downloading models')
    
    return parser.parse_args()

def is_gpu_free(gpu_id):
    if not torch.cuda.is_available():
        return False
    gpu_memory = torch.cuda.memory_reserved(gpu_id)
    return gpu_memory == 0

def run_sweep(args):
    """Setup and run a wandb sweep."""
    sweep_config = create_sweep_config(args)
    sweep_filepath = save_sweep_config(sweep_config)
    sweep_filename = os.path.basename(sweep_filepath)
    log_filename = sweep_filename.replace('.yaml', '.log')
    log_filepath = os.path.join(SWEEP_LOGS_DIR, log_filename)

    print(f"Sweep file created: {sweep_filepath}")
    
    # Start sweep
    os.system(f"wandb sweep {sweep_filepath} > {log_filepath} 2>&1")
    print(f"Log file created: {log_filepath}")
    
    # Get wandb agent command
    with open(log_filepath, 'r') as log_file:
        lines = log_file.readlines()
        if not lines:
            print("Log file is empty")
            return False
        last_line = lines[-1].strip()
    
    if "wandb agent" not in last_line:
        print("No wandb agent command found in the log file.")
        return False

    # Extract sweep ID from the agent command
    sweep_id = last_line.split()[-1]  # Get the last part of the wandb agent command which is the sweep ID
    
    # Run agents on available GPUs
    wandb_agent_command = last_line.split(": ")[-1].strip()
    print(f"wandb agent command: {wandb_agent_command}")
    
    api = wandb.Api()
    
    for gpu_id in range(2):
        runtime_log_filename = os.path.join(
            SWEEP_LOGS_DIR,
            log_filename.replace('.log', f'_gpu{gpu_id}_runtime.log')
        )
        if is_gpu_free(gpu_id):
            with open(runtime_log_filename, 'w') as runtime_log_file:
                runtime_log_file.write(f"Running on GPU{gpu_id}\n")
            os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} nohup {wandb_agent_command} > {runtime_log_filename} 2>&1 &")
            print(f"Task started on GPU{gpu_id}.")
            
        time.sleep(10)
        
        # Check sweep status using wandb API
        try:
            sweep = api.sweep(sweep_id)
            if sweep.state == "finished" or len(sweep.runs) == 0:
                break
        except Exception as e:
            print(f"Error checking sweep status: {e}")
            # Fallback to log file check if API fails
            with open(runtime_log_filename, 'r') as runtime_log_file:
                if any("Running runs: []" in line for line in runtime_log_file):
                    break

if __name__ == "__main__":
    args = parse_args()
    
    if args.use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("export HF_ENDPOINT=", os.getenv("HF_ENDPOINT"))
    
    # Convert args to dict for easier handling
    config = vars(args)
    
    # Sweep mode
    run_sweep(args)