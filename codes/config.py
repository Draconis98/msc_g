import os

# Base paths
BASE_DIR = "/home/draco/graduation"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SWEEP_LOGS_DIR = os.path.join(BASE_DIR, "sweep_logs")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

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