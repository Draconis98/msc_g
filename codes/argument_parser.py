"""Command line argument parsing module for training configuration.

This module provides functionality to parse and validate command line arguments
for model training, including strategies, model specifications, and various
training parameters.
"""

import argparse
from utils.config import (
    ALLOWED_MODELS, ALLOWED_STRATEGIES, ALLOWED_TASKS, TRAINING_DEFAULTS, TASK_TYPES
)

def parse_strategies(input_str):
    """Parse and validate comma-separated strategy list."""
    strategies = input_str.split(',')
    if not all(strategy in ALLOWED_STRATEGIES for strategy in strategies):
        raise argparse.ArgumentTypeError("Invalid strategy. Each strategy must be one of " + str(ALLOWED_STRATEGIES))
    return strategies

def parse_model(input_str):
    """Parse and validate model specifications in format 'model_name:model_size'."""
    models = input_str.split(',')
    for model in models:
        if ':' not in model:
            raise argparse.ArgumentTypeError("Invalid format. Each model must be in the format 'model_name:model_size'.")
        model_name, model_size = model.split(':')
        if model_name not in ALLOWED_MODELS or model_size not in ALLOWED_MODELS[model_name]:
            raise argparse.ArgumentTypeError("Invalid model or size. Allowed models and sizes are: " + str(ALLOWED_MODELS))
    return models

def parse_list_or_int(value, type_func=int):
    """Parse a value that can be either a single value or a comma-separated list."""
    if ',' in value:
        return [type_func(i) for i in value.split(',')]
    return type_func(value)

def setup_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Process training parameters.")
    
    # Required arguments
    parser.add_argument('-s', '--strategy', type=parse_strategies, required=True,
                       help='Strategy to use (comma-separated list)')
    parser.add_argument('-m', '--model_name', type=parse_model, required=True,
                       help='Name of the model (format: model_name:size)')
    parser.add_argument('-t', '--task', type=str, required=True,
                       choices=ALLOWED_TASKS, help='Task to perform')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--eval_dataset', type=lambda x: x.split(','), required=True,
                       help='Evaluation dataset')
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='+', required=True,
                       help='Learning rate(s)')
    parser.add_argument('-r', '--rank', type=lambda x: parse_list_or_int(x),
                       required=True, help='Rank value (comma-separated list or single integer)')
    parser.add_argument('-e', '--epochs', type=lambda x: parse_list_or_int(x),
                       required=True, help='Number of epochs (comma-separated list or single integer)')
    parser.add_argument('--target_modules', type=lambda x: x.split(','), required=True,
                       help='Target modules (comma-separated list)')
    # parser.add_argument('--modules_to_save', type=str, required=True,
    #                    choices=MODULES_TO_SAVE,
    #                    help='Modules to save (comma-separated list)')
    parser.add_argument('--task_type', type=str, required=True, choices=TASK_TYPES,
                       help='Task type, including CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, \
                           TOKEN_CLS, QUESTION_ANS and FEATURE_EXTRACTION')
    
    parser.add_argument('--data_selection', type=bool, default=False, help='Data selection method')

    # Optional arguments with defaults from config
    for key, value in TRAINING_DEFAULTS.items():
        parser.add_argument(f'--{key}', type=type(value), default=value,
                          help=f'{key.replace("_", " ").title()} (default: {value})')
    
    parser.add_argument('--use_mirror', type=bool, default=True,
                       help='Use mirror for downloading models')
    
    return parser

def parse_args():
    """Parse and return command line arguments."""
    parser = setup_parser()
    return parser.parse_args() 