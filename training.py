"""
Training pipeline for SFT.
"""

import os
import gc
from loguru import logger
import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import wandb
from peft import LoraConfig, VeraConfig, OSoraConfig, get_peft_model
from utils.config import OUTPUT_DIR
from data_processor import load_and_process_data


def cleanup_gpu_memory():
    """Clean up GPU memory by forcing garbage collection and emptying cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


class TrainingPipeline:
    """Manages the training pipeline."""
    
    def __init__(self, config, resume=False, debug=False):
        """Initialize training pipeline with configuration."""
        self.config = config
        self.output_dir = None
        self.tokenizer = None
        self.processed_dataset = None
        self.trainer = None
        self.resume = resume
        self.resume_from_checkpoint = None
        self.final_exists = False
        self.debug = debug
        
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
        name_components = [
            self.config['strategy'],
            f"seed{self.config['seed']}",
            self.config['model_name'].split('/')[-1],
            self.config['task_type'].lower(),
            self.config['dataset'].split('/')[-1],
            f"lr{self.config['learning_rate']}",
            f"{self.config['epochs']}ep",
            f"bs{self.config['batch_size']}",
            f"acc{self.config['gradient_accumulation_steps']}",
            f"warm{self.config['warmup_ratio']}",
            f"seq{self.config['max_seq_length']}",
        ]
        
        if self.config['strategy'] != 'fft':
            name_components.append(f"r{self.config['rank']}")
            if self.config['strategy'] == 'osora':
                name_components.append(f"init_osora_weights{self.config['init_osora_weights']}")
                name_components.append(f"use_dora{self.config['use_dora']}")
            elif self.config['strategy'] == 'vera':
                pass
            else:
                name_components.append(f"lora_alpha{self.config['lora_alpha']}")
            
        if self.config['packing']:
            name_components.append("packed")

        if self.config['padding_free']:
            name_components.append("padding_free")

        run_name = "-".join(name_components)

        if self.debug:
            logger.debug(f"Run name: {run_name}")

        return run_name
        
    def _setup(self):
        """Setup training environment and resources."""
        # Setup wandb run name
        try:
            if not self.resume:
                wandb.run.name = self._create_run_name()
        except Exception as e:
            logger.error(f"Failed to set wandb run name: {str(e)}")
            raise
        
        # Setup output directory
        try:
            if not self.resume:
                self.output_dir = os.path.join(
                    OUTPUT_DIR, 
                    wandb.run.name.replace("-", "/")
                )
            else:
                self.output_dir = self.config['output_dir']
            eval_dir = os.path.join(self.output_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            if self.debug:
                logger.debug(f"Evaluation directory: {eval_dir} exists.")
        except Exception as e:
            logger.error(f"Output directory setup failed: {str(e)}")
            raise
        
        logger.success("Training pipeline setup")

        if self.resume:
            final_files = ['adapter_config.json', 'adapter_model.safetensors', 'README.md']
            self.final_exists = all(os.path.exists(os.path.join(self.output_dir, f)) for f in final_files)
            if not self.final_exists:
                self.resume_from_checkpoint = self._find_latest_checkpoint()
        
    def _prepare_data(self):
        """Load and process training data."""
        logger.info(f"Loading and processing data for {self.config['dataset']}...")
        try:
            self.tokenizer, self.processed_dataset = load_and_process_data(
                self.config['model_name'], self.config['dataset'], self.config['dataset_batched'])
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def _get_training_args(self):
        """Create training arguments for the SFT trainer.

        Args:
            max_length: Truncate input sequences to reduce memory usage
        """

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
            # Note: Starting from TRL v0.16, this parameter was changed from max_seq_length to max_length
            max_length=self.config['max_seq_length'], 
            output_dir=self.output_dir,
            overwrite_output_dir=self.config['overwrite_output_dir'],
            save_steps=self.config['save_steps'],
            # save_total_limit=self.config['save_total_limit'],
            log_level="info",
            logging_steps=1,
            logging_strategy="steps",
            optim="adamw_torch",
            report_to="wandb",  # Enable wandb logging
            packing=self.config['packing'],
            padding_free=self.config['padding_free'],
            dataset_text_field="text",
        )

    def _get_lora_config(self):
        """Create LoRA configuration based on training strategy."""
        init_lora_weights = True \
            if self.config['strategy'] == 'lora' or self.config['strategy'] == 'dora' \
                else self.config['strategy']
        
        return LoraConfig(
            init_lora_weights=init_lora_weights,
            use_dora=True if self.config['strategy'] == 'dora' else False,
            r=self.config['rank'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=0.0,
            target_modules=self.config['target_modules'],
            task_type=self.config['task_type'],
        )
    
    def _get_osora_config(self):
        """Create OSORA configuration based on training strategy."""
        return OSoraConfig(
            init_osora_weights=self.config['init_osora_weights'],
            use_dora=self.config['use_dora'],
            r=self.config['rank'],
            target_modules=self.config['target_modules'],
            task_type=self.config['task_type'],
        )

    def _get_vera_config(self):
        """Create Vera configuration based on training strategy."""
        return VeraConfig(
            r=self.config['rank'],
            target_modules=self.config['target_modules'],
            task_type=self.config['task_type'],
        )
    
    def _setup_model(self):
        """Setup and configure the model for training."""
        model_kwargs = dict(
            use_cache=self.config['use_cache'],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config['bf16'] else torch.float32,
            attn_implementation=self.config['attn_implementation'],
            device_map="auto",
        )
        
        # Load the model without device mapping first
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config['model_name'], 
            **model_kwargs)
        
        # 根据策略配置PEFT模型 (Configure PEFT model based on strategy)
        if self.config['strategy'] == 'fft':
            peft_model = model
        else:
            # 策略到配置方法的映射 (Strategy to config method mapping)
            config_methods = {
                'vera': self._get_vera_config,
                'osora': self._get_osora_config
            }
            
            # 获取配置并创建PEFT模型 (Get config and create PEFT model)
            strategy = self.config['strategy']
            config_method = config_methods.get(strategy, self._get_lora_config)
            peft_config = config_method()
            peft_model = get_peft_model(model, peft_config)
            
            # 记录配置到wandb (Log config to wandb)
            if not self.resume:
                config_name = f"{strategy}_config" if strategy in config_methods else "lora_config"
                wandb.config.update({config_name: peft_config.__dict__})
        
        return peft_model
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint directory in the output_dir, return None if not found."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return None
        checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith('checkpoint-')]
        if not checkpoints:
            return None
        # Get the largest step number
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else -1)
        return os.path.join(self.output_dir, latest_ckpt)
    
    def _setup_trainer(self):
        """Setup the trainer."""
        logger.info("Setting up trainer...")
        # Setup model
        model = self._setup_model()

        # if self.debug:
        # for name, param in model.named_parameters():
            # logger.debug(f"{name} requires_grad: {param.requires_grad} shape: {param.shape}")
        
        # Setup training arguments
        training_args = self._get_training_args()
        
        self.trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.processed_dataset,
        )

        logger.success("Trainer setup completed")
        
    def train(self):
        """Run the training process."""
        logger.info("Starting training...")
        try:
            if self.resume_from_checkpoint:
                logger.info(f"Resume from checkpoint: {self.resume_from_checkpoint}")
            train_result = self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            metrics = train_result.metrics
            
            # Log metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            # Save model and tokenizer
            self.trainer.model.save_pretrained(self.output_dir, save_config=True)
            self.tokenizer.save_pretrained(self.output_dir, save_config=True)
            
            logger.success("Training completed")
            return metrics
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            wandb.log({"training_error": str(e)})
            raise
        
    def run(self):
        """Run the complete training pipeline."""
        self._setup()
        if not self.final_exists:
            self._prepare_data()
            self._setup_trainer()
            self.train()
        else:
            logger.info("Final step exists, skip training")