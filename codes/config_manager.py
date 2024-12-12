import wandb
import logging

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