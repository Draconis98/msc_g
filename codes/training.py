import os
from loguru import logger
import wandb
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from utils.misc import ensure_dir, get_output_dir
from utils.gpu import cleanup_gpu_memory
from data_processor import load_and_process_data, get_model_checkpoint

class TrainingPipeline:
    """Manages the training pipeline."""
    
    def __init__(self, config):
        """Initialize training pipeline with configuration."""
        self.config = config
        self.output_dir = None
        self.tokenizer = None
        self.processed_dataset = None
        self.trainer = None
        
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'model'):
                # Delete the model explicitly
                del self.trainer.model
            del self.trainer
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            
        if hasattr(self, 'processed_dataset') and self.processed_dataset is not None:
            del self.processed_dataset

        if hasattr(self, 'output_dir') and self.output_dir is not None:
            del self.output_dir
            
        if hasattr(self, 'config') and self.config is not None:
            del self.config

        # clear gpu memory
        cleanup_gpu_memory()
    
    def _create_run_name(self):
        """Create a standardized run name from configuration."""
        if self.config['data_selection']:
            return f"{self.config['strategy']}-{self.config['model_name']}-{self.config['task']}-" \
                f"{self.config['dataset']}-{self.config['learning_rate']}-r{self.config['rank']}-" \
                f"{self.config['epochs']}epochs-ds"
        else:
            return f"{self.config['strategy']}-{self.config['model_name']}-{self.config['task']}-" \
                f"{self.config['dataset']}-{self.config['learning_rate']}-r{self.config['rank']}-" \
                f"{self.config['epochs']}epochs"
        
    def _setup(self):
        """Setup training environment and resources."""
        # Setup wandb run name
        try:
            wandb.run.name = self._create_run_name()
        except Exception as e:
            logger.error("Failed to set wandb run name: %s", str(e))
            raise
        
        # Setup output directory
        try:
            self.output_dir = get_output_dir(self.config)
            ensure_dir(os.path.join(self.output_dir, "eval"))
        except Exception as e:
            logger.error("Failed to setup output directory: %s", str(e))
            raise
        
        logger.success("Training pipeline setup completed")
        
    def _prepare_data(self):
        """Load and process training data."""
        logger.info("Loading and processing data for %s...", self.config['dataset'])
        try:
            self.tokenizer, self.processed_dataset = load_and_process_data(self.config)
        except Exception as e:
            logger.error("Failed to load and process data: %s", str(e))
            raise
        logger.success("Data preparation completed")

    def _get_training_args(self):
        """Create training arguments for the SFT trainer."""
        return SFTConfig(
            bf16=self.config['bf16'],
            learning_rate=self.config['learning_rate'],
            lr_scheduler_type=self.config['learning_schedule'],
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_checkpointing=self.config['gradient_checkpointing'],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            warmup_ratio=self.config['warmup_ratio'],
            max_seq_length=self.config['max_seq_length'],
            output_dir=self.output_dir,
            overwrite_output_dir=self.config['overwrite_output_dir'],
            save_steps=self.config['save_steps'],
            save_total_limit=self.config['save_total_limit'],
            log_level="info",
            logging_steps=1,
            logging_strategy="steps",
            optim="adamw_torch",
            report_to="wandb",  # Enable wandb logging
            packing=self.config['packing'],
            dataset_text_field="text",
        )

    def _get_lora_config(self):
        """Create LoRA configuration based on training strategy."""
        init_lora_weights = self.config['strategy'] if self.config['strategy'].startswith(('dude', 'pissa')) else True
        
        return LoraConfig(
            init_lora_weights=init_lora_weights,
            use_dora=False if self.config['strategy'] in ['lora', 'pissa'] else True,
            r=self.config['rank'],
            lora_alpha=self.config['rank'] if self.config['strategy'].startswith(('dude', 'pissa')) else 2*self.config['rank'],
            lora_dropout=0.0,
            target_modules=self.config['target_modules'],
            modules_to_save=['lm_head', 'embed_token'],
            task_type=self.config['task_type'],
        )

    def _setup_model(self):
        """Setup and configure the model for training."""
        checkpoint_path = get_model_checkpoint(self.config['model_name'])
        model_kwargs = dict(
            use_cache=self.config['use_cache'],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config['bf16'] else torch.float16,
            device_map="auto",
        )
        
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
        
        if self.config['strategy'] != 'fft':
            lora_config = self._get_lora_config()
            model = get_peft_model(model, lora_config)
            # Log LoRA configuration to wandb
            wandb.config.update({"lora_config": lora_config.__dict__})
        
        return model
        
    def _setup_trainer(self):
        """Setup the trainer."""
        logger.info("Setting up trainer...")
        # Setup model
        model = self._setup_model()
        
        # Setup training arguments
        training_args = self._get_training_args()
        
        self.trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.processed_dataset,
            tokenizer=self.tokenizer,
        )
        logger.success("Trainer setup completed")
        
    def train(self):
        """Run the training process."""
        logger.info("Starting training...")
        try:
            train_result = self.trainer.train()
            metrics = train_result.metrics
            
            # Log metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            # Save model and tokenizer
            self.trainer.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.success("Training completed")
            return metrics
        
        except Exception as e:
            wandb.log({"training_error": str(e)})
            raise e
        
    def run(self):
        """Run the complete training pipeline."""
        self._setup()
        self._prepare_data()
        self._setup_trainer()
        self.train()