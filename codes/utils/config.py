"""Configuration management module for the training pipeline."""

import os
import logging
import wandb

# Base paths
BASE_DIR = "/home/draco/graduation"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SWEEP_LOGS_DIR = os.path.join(BASE_DIR, "sweep_logs")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

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

# Model configurations
ALLOWED_MODELS = {
    'llama2': ['7b', '13b'],
    'llama3': ['8b'],
    'qwen1.5': ['1.8b', '4b', '7b', '14b', '32b'],
    'qwen2': ['0.5b', '1.5b', '7b'],
    'qwen2.5': ['0.5b', '1.5b', '3b', '7b', '14b', '32b'],
    'qwq': ['32b'],
    'mistral': ['7b-v0.1', '7b-v0.2', '7b-v0.3', 'nemo', 'small'],
    'gemma': ['2b', '7b'],
    'gemma1.1': ['2b', '7b'],
    'gemma2': ['2b', '9b', '27b'],
    'phi3': ['mini', 'small', 'medium'],
    'phi3.5': ['mini'],
    'phi4': ['small'],
    't5': ['small', 'base', 'large'],
}

# Training strategies
ALLOWED_STRATEGIES = {'fft', 'lora', 'dora', 'pissa', 'dude'}

# Tasks and datasets
ALLOWED_TASKS = ['math', 'code', 'commonsenseqa', 'mmlu', 'super_glue', 'mmlupro', 'gpqa', 'truthfulqa', 'openbookqa', 'agieval']
TASK_TYPES = ['CAUSAL_LM', 'SEQ_CLS', 'SEQ_2_SEQ_LM', 'TOKEN_CLS', 'QUESTION_ANS', 'FEATURE_EXTRACTION']

# MODULES_TO_SAVE = ['lm_head,embed_token', 'lm_head', 'embed_token']

# OpenCompass configuration
OPENCOMPASS_DIR = os.path.join(BASE_DIR, "codes/oc")

COMMON_PATH = "/home/draco/graduation/models"

MODEL_PATHS = {
    'llama2': {
        '7b': f"{COMMON_PATH}/Llama/Llama-2-7b-chat-hf",
        '13b': f"{COMMON_PATH}/Llama/Llama-2-13b-chat-hf"
    },
    'llama3': {
        '8b': f"{COMMON_PATH}/Llama/Meta-Llama-3-8B-Instruct"
    },
    'qwen1.5': {
        '1.8b': f"{COMMON_PATH}/Qwen/Qwen1.5-1.8B-Chat",
        '4b': f"{COMMON_PATH}/Qwen/Qwen1.5-4B-Chat",
        '7b': f"{COMMON_PATH}/Qwen/Qwen1.5-7B-Chat",
        '14b': f"{COMMON_PATH}/Qwen/Qwen1.5-14B-Chat",
        '32b': f"{COMMON_PATH}/Qwen/Qwen1.5-32B-Chat"
    },
    'qwen2': {
        '0.5b': f"{COMMON_PATH}/Qwen/Qwen2-0.5B-Instruct",
        '1.5b': f"{COMMON_PATH}/Qwen/Qwen2-1.5B-Instruct",
        '7b': f"{COMMON_PATH}/Qwen/Qwen2-7B-Instruct"
    },
    'qwen2.5': {
        '0.5b': f"{COMMON_PATH}/Qwen/Qwen2.5-0.5B-Instruct",
        '1.5b': f"{COMMON_PATH}/Qwen/Qwen2.5-1.5B-Instruct",
        '3b': f"{COMMON_PATH}/Qwen/Qwen2.5-3B-Instruct",
        '7b': f"{COMMON_PATH}/Qwen/Qwen2.5-7B-Instruct",
        '14b': f"{COMMON_PATH}/Qwen/Qwen2.5-14B-Instruct",
        '32b': f"{COMMON_PATH}/Qwen/Qwen2.5-32B-Instruct"
    },
    'qwq': {
        '32b': f"{COMMON_PATH}/Qwen/QwQ-32B-Preview"
    },
    'mistral': {
        '7b-v0.1': f"{COMMON_PATH}/Mistral/Mistral-7B-Instruct-v0.1",
        '7b-v0.2': f"{COMMON_PATH}/Mistral/Mistral-7B-Instruct-v0.2",
        '7b-v0.3': f"{COMMON_PATH}/Mistral/Mistral-7B-Instruct-v0.3",
        'nemo': f"{COMMON_PATH}/Mistral/Mistral-Nemo-Instruct-2407",
        'small': f"{COMMON_PATH}/Mistral/Mistral-Small-Instruct-2409"
    },
    'gemma': {
        '2b': f"{COMMON_PATH}/Gemma/gemma-2b-it",
        '7b': f"{COMMON_PATH}/Gemma/gemma-7b-it"
    },
    'gemma1.1': {
        '2b': f"{COMMON_PATH}/Gemma/gemma-1.1-2b-it",
        '7b': f"{COMMON_PATH}/Gemma/gemma-1.1-7b-it"
    },
    'gemma2': {
        '2b': f"{COMMON_PATH}/Gemma/gemma-2b-it",
        '9b': f"{COMMON_PATH}/Gemma/gemma-2-9b-it",
        '27b': f"{COMMON_PATH}/Gemma/gemma-2-27b-it"
    },
    'phi3': {
        'mini': f"{COMMON_PATH}/Phi/Phi-3-mini-128k-instruct",
        'small': f"{COMMON_PATH}/Phi/Phi-3-small-128k-instruct",
        'medium': f"{COMMON_PATH}/Phi/Phi-3-medium-128k-instruct"
    },
    'phi3.5': {
        'mini': f"{COMMON_PATH}/Phi/Phi-3.5-mini-instruct"
    },
    'phi4': {
        'small': f"{COMMON_PATH}/Phi/phi-4"
    },
    't5': {
        'small': f"{COMMON_PATH}/google-t5/t5-small",
        'base': f"{COMMON_PATH}/google-t5/t5-base",
        'large': f"{COMMON_PATH}/google-t5/t5-large"
    }
}

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
            'task_type': wandb.config.task_type,
            
            # Saving configuration
            'save_steps': wandb.config.save_steps,
            'save_total_limit': wandb.config.save_total_limit,
            'overwrite_output_dir': wandb.config.overwrite_output_dir,
            # 'modules_to_save': wandb.config.modules_to_save,
            
            # Add seed configuration
            'seed': wandb.config.seed,
        }
        
        logging.info("Configuration created successfully")
        return config
    
    @staticmethod
    def validate_config(config):
        """Validate configuration parameters."""
        required_fields = [
            'model_name', 'strategy', 'task', 'dataset', 'eval_dataset',
            'learning_rate', 'epochs', 'batch_size', 'task_type'
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
        
        logging.info("Configuration validated successfully")
        return True
    
    @staticmethod
    def log_config(config):
        """Log configuration parameters."""
        logging.info("Current configuration:")
        for key, value in config.items():
            logging.info("  %s: %s", key, value)
            
    @staticmethod
    def update_config(config, updates):
        """Update configuration with new values."""
        for key, value in updates.items():
            if key in config:
                config[key] = value
                logging.info("Updated %s to %s", key, value)
            else:
                logging.warning("Attempted to update non-existent config key: %s", key)
        return config 