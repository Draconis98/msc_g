import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from data_processor import get_model_checkpoint
import wandb
import os
from utils.config import OUTPUT_DIR

def get_output_dir(config):
    """Get the output directory for the training."""
    return os.path.join(
        OUTPUT_DIR,
        f"{config['strategy']}/{config['task']}/{config['dataset']}/" \
        f"{config['model_name'].replace(':', '/')}/" \
        f"{config['learning_rate']}/r{config['rank']}/{config['epochs']}epochs"
    )

def get_training_args(config, output_dir):
    """Create training arguments for the SFT trainer."""
    return SFTConfig(
        bf16=config['bf16'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['learning_schedule'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_ratio=config['warmup_ratio'],
        max_seq_length=config['max_seq_length'],
        output_dir=output_dir,
        overwrite_output_dir=config['overwrite_output_dir'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        log_level="info",
        logging_steps=1,
        logging_strategy="steps",
        optim="adamw_torch",
        report_to="wandb",  # Enable wandb logging
        packing=config['packing'],
        dataset_text_field="text",
    )

def get_lora_config(config):
    """Create LoRA configuration based on training strategy."""
    init_lora_weights = config['strategy'] if config['strategy'].startswith(('dude', 'pissa')) else True
    
    return LoraConfig(
        init_lora_weights=init_lora_weights,
        use_dora=False if config['strategy'] in ['lora', 'pissa'] else True,
        r=config['rank'],
        lora_alpha=config['rank'] if config['strategy'].startswith(('dude', 'pissa')) else 2*config['rank'],
        lora_dropout=0.0,
        target_modules=config['target_modules'],
    )

def setup_model(config):
    """Setup and configure the model for training."""
    checkpoint_path = get_model_checkpoint(config['model_name'])
    model_kwargs = dict(
        use_cache=config['use_cache'],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config['bf16'] else torch.float16,
        device_map="auto",
    )
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    
    if config['strategy'] != 'fft':
        lora_config = get_lora_config(config)
        model = get_peft_model(model, lora_config)
        # Log LoRA configuration to wandb
        wandb.config.update({"lora_config": lora_config.__dict__})
    
    return model

def setup_training(config, tokenizer, processed_dataset, output_dir):
    """Setup complete training environment."""
    # Set random seeds for reproducibility
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
    
    # Setup model
    model = setup_model(config)
    
    # Setup training arguments
    training_args = get_training_args(config, output_dir)
    
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )

def train_model(trainer, output_dir, tokenizer):
    """Execute model training and save results."""
    try:
        train_result = trainer.train()
        metrics = train_result.metrics
        
        # Log metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save model and tokenizer
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return metrics
    
    except Exception as e:
        wandb.log({"training_error": str(e)})
        raise e