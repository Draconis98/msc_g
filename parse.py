"""Command line argument parsing module for training and evaluation configuration.

This module provides functionality to parse and validate command line arguments
for model training, including strategies, model specifications, and various
training and evaluation parameters.
"""

import argparse

from utils.config import TRAINING_DEFAULTS, TASK_TYPES


def parse_str2int_list(input_str):
    """Parse a string to a list of integers.
    
    Args:
        input_str (str): The input string to parse.
    
    Returns:
        list: A list of integers.
    """
    return [int(i) for i in input_str.split(',')]

def parse_str2float_list(input_str):
    """Parse a string to a list of floats.
    
    Args:
        input_str (str): The input string to parse.
    
    Returns:
        list: A list of floats.
    """
    return [float(i) for i in input_str.split(',')]

def setup_parser():
    """Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process training parameters.")
    
    # Required arguments
    parser.add_argument('-s', '--strategy', type=lambda x: x.split(','), required=True,
                       help='Strategy to use (comma-separated list, e.g. fft,lora)')
    parser.add_argument('-m', '--model_name', type=lambda x: x.split(','), required=True,
                       help='Name of the model (comma-separated list, e.g. Qwen/Qwen3-0.6B,Qwen/Qwen2-0.5B)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='Name of the dataset (format: huggingface_dataset_name, e.g. cais/mmlu)')
    parser.add_argument('-lr', '--learning_rate', type=parse_str2float_list, required=True,
                       help='Learning rate(s) (comma-separated list or single float, e.g. 1e-4,1e-5)')
    parser.add_argument('-ed', '--eval_dataset', type=lambda x: x.split(','), required=True,
                       help='Name of the evaluation dataset (comma-separated list, e.g. mmlu_gen,gsm8k_gen)')
    parser.add_argument('-e', '--epochs', type=parse_str2int_list, required=True,
                       help='Number of epochs (comma-separated list or single integer, e.g. 1,2)')
    parser.add_argument('-tt', '--task_type', type=str, default=TASK_TYPES[0], choices=TASK_TYPES,
                       help='Task type, including CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, \
                           TOKEN_CLS, QUESTION_ANS and FEATURE_EXTRACTION')
    
    # LoRA: Low-Rank Adaptation
    parser.add_argument('-r', '--rank', type=parse_str2int_list, required=True, 
                       help='Rank value (comma-separated list or single integer, e.g. 16,32)')
    parser.add_argument('-t', '--target_modules', type=lambda x: x.split(','), required=True,
                       help='Target modules (comma-separated list, e.g. q_proj,k_proj,v_proj)')
    
    # WandB
    parser.add_argument('--wandb_project', type=str, required=True,
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, required=True,
                       help='WandB entity name')
    
    parser.add_argument('--attn_implementation', type=str, default='sdpa',
                       help='Attention implementation, including sdpa, flash_attention_2 and eager')
    parser.add_argument('--enable_thinking', type=bool, default=False,
                       help='Enable thinking for evaluation')
    
    # Optional arguments with defaults from config
    for key, value in TRAINING_DEFAULTS.items():
        parser.add_argument(f'--{key}', type=type(value), default=value,
                          help=f'{key.replace("_", " ").title()} (default: {value})')

    parser.add_argument('--use_mirror', type=bool, default=True,
                       help='Use mirror for downloading models')
    
    parser.add_argument('--debug', type=bool, default=False,
                       help='Debug mode')
    
    return parser

def parse_args():
    """Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = setup_parser()
    return parser.parse_args() 