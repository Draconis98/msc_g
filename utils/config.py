"""Configuration management module for the training pipeline."""

import os
from loguru import logger
import wandb

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OPENCOMPASS_DIR = os.path.join(BASE_DIR, "opencompass")

# Training defaults
TRAINING_DEFAULTS = {
    "learning_schedule": "cosine",
    "save_steps": 100,
    "save_total_limit": 8,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 128,
    "packing": False,
    "padding_free": False,
    "max_seq_length": 4096,
    "overwrite_output_dir": True,
    "bf16": True,
    "use_cache": False,
    "dataset_batched": False,
}

# Task types
TASK_TYPES = ['CAUSAL_LM', 'SEQ_CLS', 'SEQ_2_SEQ_LM', 'TOKEN_CLS', \
              'QUESTION_ANS', 'FEATURE_EXTRACTION']

class ConfigManager:
    """Manages configuration for the training pipeline."""
    
    @staticmethod
    def create_config():
        """Create configuration dictionary from wandb config."""
        config = {
            # Model configuration
            'strategy': wandb.config.strategy,
            'model_name': wandb.config.model_name,
            'bf16': wandb.config.bf16,
            'use_cache': wandb.config.use_cache,
            'attn_implementation': wandb.config.attn_implementation,

            # Task and dataset configuration
            'dataset': wandb.config.dataset,
            'eval_dataset': wandb.config.eval_dataset,
            'dataset_batched': wandb.config.dataset_batched,

            # Training configuration
            'epochs': wandb.config.epochs,
            'batch_size': wandb.config.batch_size,
            'learning_rate': wandb.config.learning_rate,
            'learning_schedule': wandb.config.learning_schedule,
            'gradient_checkpointing': wandb.config.gradient_checkpointing,
            'gradient_accumulation_steps': wandb.config.gradient_accumulation_steps,
            'warmup_ratio': wandb.config.warmup_ratio,
            'packing': wandb.config.packing,
            'padding_free': wandb.config.padding_free,
            'max_seq_length': wandb.config.max_seq_length,
            'task_type': wandb.config.task_type,
            
            # Saving configuration
            'save_steps': wandb.config.save_steps,
            'save_total_limit': wandb.config.save_total_limit,
            'overwrite_output_dir': wandb.config.overwrite_output_dir,

            # Evaluation configuration
            'enable_thinking': wandb.config.enable_thinking,
            'max_out_len': wandb.config.max_out_len,
            
            # Add seed configuration
            'seed': wandb.config.seed,
        }

        if wandb.config.strategy != 'fft':
            for param in ['target_modules', 'rank']:
                config[param] = getattr(wandb.config, param)
        
        logger.success("Configuration created successfully")
        return config