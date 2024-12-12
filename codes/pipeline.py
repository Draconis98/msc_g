import wandb
import os
import sys
import logging
import random
import numpy as np
import torch
import csv
from trainer import setup_training, train_model, get_output_dir
from data_processor import load_and_process_data
from evaluator import setup_evaluator
from config import OPENCOMPASS_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    # Initialize wandb
    wandb.init()
    
    # Get configuration
    config = {
        # Model configuration
        'model_name': wandb.config.model_name,
        'strategy': wandb.config.strategy,
        'rank': wandb.config.rank,
        'target_modules': wandb.config.target_modules,
        'bf16': wandb.config.bf16,
        'use_cache': wandb.config.use_cache,
        
        # Task and dataset configuration
        'task': wandb.config.task,
        'dataset': wandb.config.dataset,
        'eval_dataset': wandb.config.eval_dataset,
        'dataset_batched': wandb.config.dataset_batched,
        
        # Training configuration
        'epochs': wandb.config.epochs,
        'batch_size': wandb.config.batch_size,
        'learning_rate': wandb.config.learning_rate if wandb.config.strategy != 'fft' 
                        else wandb.config.learning_rate / 10,
        'learning_schedule': wandb.config.learning_schedule,
        'gradient_checkpointing': wandb.config.gradient_checkpointing,
        'gradient_accumulation_steps': wandb.config.gradient_accumulation_steps,
        'warmup_ratio': wandb.config.warmup_ratio,
        'packing': wandb.config.packing,
        'max_seq_length': wandb.config.max_seq_length,
        
        # Saving configuration
        'save_steps': wandb.config.save_steps,
        'save_total_limit': wandb.config.save_total_limit,
        'overwrite_output_dir': wandb.config.overwrite_output_dir,
        
        # Add seed configuration
        'seed': wandb.config.seed,
    }
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup wandb run name
    wandb.run.name = "{}-{}-{}-{}-{}-r{}-{}epochs-seed{}".format(
        config['strategy'], config['model_name'], config['task'], config['dataset'],
        config['learning_rate'], config['rank'], config['epochs'], config['seed']
    )
    
    # Setup output directory and logging
    output_dir = get_output_dir(config)
    os.makedirs(os.path.join(output_dir, "eval"), exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "peft.log")
    sys.stdout = open(log_file, "w")
    sys.stderr = open(log_file, "w")
    
    # Load and process data
    logging.info(f"Loading and processing data for {config['dataset']}...")
    tokenizer, processed_dataset = load_and_process_data(config)
    
    # Setup and run training
    logging.info(f"Setting up training...")
    trainer = setup_training(config, tokenizer, processed_dataset, output_dir)
    train_model(trainer, output_dir, tokenizer)
    
    # Setup and run evaluation
    logging.info(f"Setting up evaluation...")
    evaluator = setup_evaluator(config, output_dir)
    logging.info(f"Running evaluation...")
    opencompass_run = os.path.join(OPENCOMPASS_DIR, "run.py")
    eval_cmd = f"python {opencompass_run} --models {evaluator} --datasets {config['eval_dataset']}_gen -w {output_dir}/eval"
    os.system(eval_cmd)
    
    # Get the latest evaluation folder
    eval_dir = f"{output_dir}/eval"
    eval_runs = [d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))]
    latest_run = max(eval_runs, key=lambda x: os.path.getctime(os.path.join(eval_dir, x)))
    summary_dir = os.path.join(eval_dir, latest_run, "summary")
    
    # Parse results from summary files
    for filename in os.listdir(summary_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(summary_dir, filename)
            with open(filepath, 'r') as f:
                # Parse CSV file
                reader = csv.DictReader(f)
                # Get the last column name (model name)
                fieldnames = reader.fieldnames
                model_column = fieldnames[-1]  # Get the last column name
                
                for row in reader:
                    try:
                        dataset = row['dataset']
                        version = row['version']
                        metric = row['metric']
                        mode = row['mode']
                        value = float(row[model_column])
                        wandb.log({f"eval/{dataset}/{version}/{metric}/{mode}": value})
                    except KeyError as e:
                        logging.warning(f"Missing expected column: {e}")
                    except ValueError as e:
                        logging.warning(f"Could not parse value: {e}")
    
    logging.info(f"Evaluation results logged to wandb from {summary_dir}")
    
    # Cleanup
    wandb.finish()
    sys.stdout.close()
    sys.stderr.close()

if __name__ == "__main__":
    main()