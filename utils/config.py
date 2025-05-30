"""Configuration management module for the training pipeline."""

import os
from loguru import logger
import wandb

# Base paths
BASE_DIR = "/home/draco/codes"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Wandb configuration
WANDB_CONFIG = {
    "project": "graduation-project",
    "entity": "draco98",
}

# Training defaults
TRAINING_DEFAULTS = {
    "learning_schedule": "cosine",
    "batch_size": 1,
    "save_steps": 100,
    "save_total_limit": 8,
    "gradient_checkpointing": True,
    "gradient_accumulation_steps": 128,
    "warmup_ratio": 0.03,
    "packing": False,
    "max_seq_length": 4096,
    "overwrite_output_dir": True,
    "bf16": True,
    "use_cache": False,
    "dataset_batched": False,
    "seed": 42,
}

# Task types
TASK_TYPES = ['CAUSAL_LM', 'SEQ_CLS', 'SEQ_2_SEQ_LM', 'TOKEN_CLS', 'QUESTION_ANS', 'FEATURE_EXTRACTION']

# OpenCompass configuration
OPENCOMPASS_DIR = os.path.join(BASE_DIR, "opencompass")

class ConfigManager:
    """Manages configuration for the training pipeline."""
    
    @staticmethod
    def create_config():
        """Create configuration dictionary from wandb config."""
        config = {
            # Model configuration
            'model_name': wandb.config.model_name,
            'strategy': wandb.config.strategy,
            'rank': wandb.config.rank,
            'target_modules': wandb.config.target_modules,
            'bf16': wandb.config.bf16,
            'use_cache': wandb.config.use_cache,
            
            # Task and dataset configuration
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
            'task_type': wandb.config.task_type,
            
            # Saving configuration
            'save_steps': wandb.config.save_steps,
            'save_total_limit': wandb.config.save_total_limit,
            'overwrite_output_dir': wandb.config.overwrite_output_dir,
            
            # Add seed configuration
            'seed': wandb.config.seed,
        }
        
        logger.success("Configuration created successfully")
        return config
    
    @staticmethod
    def validate_config(config):
        """Validate configuration parameters."""
        required_fields = [
            'model_name', 'strategy', 'dataset', 'eval_dataset',
            'learning_rate', 'epochs', 'batch_size'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
            
        if config['learning_rate'] <= 0:
            raise ValueError("Learning rate must be positive")
            
        if config['epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")
            
        if config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        
        logger.success("Configuration validated successfully")
        return True
    
    @staticmethod
    def log_config(config):
        """Log configuration parameters."""
        logger.info("Current configuration:")
        for key, value in config.items():
            logger.info("  %s: %s", key, value)
            
    @staticmethod
    def update_config(config, updates):
        """Update configuration with new values."""
        for key, value in updates.items():
            if key in config:
                config[key] = value
                logger.info("Updated %s to %s", key, value)
            else:
                logger.warning("Attempted to update non-existent config key: %s", key)
        return config 