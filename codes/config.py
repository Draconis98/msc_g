import os

# Base paths
BASE_DIR = "/home/draco/graduation"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SWEEP_LOGS_DIR = os.path.join(BASE_DIR, "sweep_logs")

# Wandb configuration
WANDB_CONFIG = {
    "project": "graduation-project",
    "entity": "draco98h",
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
    'mistral': ['7b'],
    'gemma': ['2b'],
    'gemma2': ['9b'],
    'phi3': ['3.8b']
}

# Training strategies
ALLOWED_STRATEGIES = {'fft', 'lora', 'dora', 'pissa', 'dude_0.0'}

# Tasks and datasets
ALLOWED_TASKS = ['math', 'code', 'commonsense', 'mmlu', 'super_glue']

# OpenCompass configuration
OPENCOMPASS_DIR = os.path.join(BASE_DIR, "codes/oc") 