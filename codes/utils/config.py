import os
import wandb
import logging

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
    'qwen1.5': ['1.8b', '4b', '7b', '14b'],
    'qwen2': ['0.5b', '1.5b', '7b'],
    'qwen2.5': ['0.5b'],
    'mistral': ['7b'],
    'gemma': ['2b'],
    'gemma2': ['9b'],
    'phi3': ['3.8b']
}

# Training strategies
ALLOWED_STRATEGIES = {'fft', 'lora', 'dora', 'pissa', 'dude'}

# Tasks and datasets
ALLOWED_TASKS = ['math', 'code', 'commonsense', 'mmlu', 'super_glue']

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
        '14b': f"{COMMON_PATH}/Qwen/Qwen1.5-14B-Chat"
    },
    'qwen2': {
        '0.5b': f"{COMMON_PATH}/Qwen/Qwen2-0.5B-Instruct",
        '1.5b': f"{COMMON_PATH}/Qwen/Qwen2-1.5B-Instruct",
        '7b': f"{COMMON_PATH}/Qwen/Qwen2-7B-Instruct"
    },
    'qwen2.5': {
        '0.5b': f"{COMMON_PATH}/Qwen/Qwen2.5-0.5B-Instruct"
    },
    'mistral': {
        '7b': f"{COMMON_PATH}/Mistral/Mistral-7B-Instruct-v0.2"
    },
    'gemma': {
        '2b': f"{COMMON_PATH}/Gemma/gemma-2b-it"
    },
    'gemma2': {
        '9b': f"{COMMON_PATH}/Gemma/gemma-2-9b-it"
    },
    'phi3': {
        '3.8b': f"{COMMON_PATH}/Phi/Phi-3-mini-128k-instruct"
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
            
            # Saving configuration
            'save_steps': wandb.config.save_steps,
            'save_total_limit': wandb.config.save_total_limit,
            'overwrite_output_dir': wandb.config.overwrite_output_dir,
            
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
        
        logging.info("Configuration validated successfully")
        return True
    
    @staticmethod
    def log_config(config):
        """Log configuration parameters."""
        logging.info("Current configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")
            
    @staticmethod
    def update_config(config, updates):
        """Update configuration with new values."""
        for key, value in updates.items():
            if key in config:
                config[key] = value
                logging.info(f"Updated {key} to {value}")
            else:
                logging.warning(f"Attempted to update non-existent config key: {key}")
        return config 