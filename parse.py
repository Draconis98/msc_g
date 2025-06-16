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
    
    # Basic group
    basic_group = parser.add_argument_group('Basic', 'Basic configuration parameters')
    basic_group.add_argument('--use_mirror', type=bool, default=True,
                       help='Use mirror for downloading models')
    basic_group.add_argument('--seed', type=parse_str2int_list, default=42,
                       help='Random seed (comma-separated list or single integer, e.g. 42,123)')
    basic_group.add_argument('--debug', type=bool, default=False,
                       help='Debug mode')

    # Resume group
    resume_group = parser.add_argument_group('Resume', 'Training resume configuration')
    resume_group.add_argument('--resume', type=bool, default=False,
                       help='Resume training and evaluation')
    resume_group.add_argument('--run_id', type=str, default=None,
                       help='W&B Run ID')

    # WandB group
    wandb_group = parser.add_argument_group('WandB', 'Weights & Biases configuration')
    wandb_group.add_argument('--wandb_project', type=str,
                       help='WandB project name')
    wandb_group.add_argument('--wandb_entity', type=str,
                       help='WandB entity name')
    wandb_group.add_argument('--sweep_method', type=str, default='grid', choices=['grid', 'random', 'bayes'],
                       help='Sweep method, including grid, random and bayes')
    wandb_group.add_argument('--sweep_metric', type=str, default='eval_accuracy',
                       help='Sweep metric, including eval_loss, eval_accuracy, eval_f1, eval_rouge, eval_bleu, \
                         eval_meteor, eval_bertscore, eval_rouge_l, eval_rouge_l_summary, eval_rouge_l_summary_f1, \
                         eval_rouge_l_summary_recall, eval_rouge_l_summary_precision')
    wandb_group.add_argument('--sweep_count', type=int, default=10,
                       help='Maximum number of runs in sweep (default: 10)')

    # Training group
    training_group = parser.add_argument_group('Training', 'Model training configuration')
    training_group.add_argument('-s', '--strategy', type=lambda x: x.split(','),
                       help='Strategy to use (comma-separated list, e.g. fft,lora)')
    training_group.add_argument('-m', '--model_name', type=lambda x: x.split(','),
                       help='Name of the model (comma-separated list, e.g. Qwen/Qwen3-0.6B,Qwen/Qwen2-0.5B)')
    training_group.add_argument('-d', '--dataset', type=str,
                       help='Name of the dataset (format: huggingface_dataset_name, e.g. cais/mmlu)')
    training_group.add_argument('-lr', '--learning_rate', type=parse_str2float_list,
                       help='Learning rate(s) (comma-separated list or single float, e.g. 1e-4,1e-5)')
    training_group.add_argument('-e', '--epochs', type=parse_str2int_list,
                       help='Number of epochs (comma-separated list or single integer, e.g. 1,2)')
    training_group.add_argument('-tt', '--task_type', type=str, default=TASK_TYPES[0], choices=TASK_TYPES,
                       help='Task type, including CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, TOKEN_CLS, QUESTION_ANS and FEATURE_EXTRACTION')
    training_group.add_argument('--attn_implementation', type=str, default='sdpa',
                       help='Attention implementation, including sdpa, flash_attention_2 and eager')
    training_group.add_argument('-bs', '--batch_size', type=parse_str2int_list, default=1,
                       help='Batch size(s) (comma-separated list or single integer, e.g. 1,2)')
    training_group.add_argument('--warmup_ratio', type=parse_str2float_list, default=0.03,
                       help='Warmup ratio(s) (comma-separated list or single float, e.g. 0.03,0.05)')

    # Evaluation group
    eval_group = parser.add_argument_group('Evaluation', 'Model evaluation configuration')
    eval_group.add_argument('-ed', '--eval_dataset', type=lambda x: x.split(','),
                       help='Name of the evaluation dataset (comma-separated list, e.g. mmlu_gen,gsm8k_gen)')
    eval_group.add_argument('--enable_thinking', type=bool, default=False,
                       help='Enable thinking for evaluation')
    eval_group.add_argument('--max_out_len', type=parse_str2int_list, default=512,
                       help='Maximum output length for evaluation, e.g. 512')

    # LoRA group
    lora_group = parser.add_argument_group('LoRA', 'Low-Rank Adaptation parameters')
    lora_group.add_argument('-r', '--rank', type=parse_str2int_list,
                       help='Rank value (comma-separated list or single integer, e.g. 16,32)')
    lora_group.add_argument('-t', '--target_modules', type=lambda x: x.split(','),
                       help='Target modules (comma-separated list, e.g. q_proj,k_proj,v_proj)')

    # Default training group
    default_group = parser.add_argument_group('Defaults', 'Optional training parameters with default values')
    for key, value in TRAINING_DEFAULTS.items():
        default_group.add_argument(f'--{key}', type=type(value), default=value,
                          help=f'{key.replace("_", " ").title()} (default: {value})')

    return parser

def parse_args():
    """Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate resume mode requirements
    if args.resume:
        if args.run_id is None:
            raise ValueError("Run ID is required when resuming")
    else:
        # Validate required parameters for non-resume mode
        required_params = {
            'wandb_project': 'WandB project name',
            'wandb_entity': 'WandB entity name',
            'strategy': 'Strategy',
            'model_name': 'Model name',
            'dataset': 'Dataset',
            'learning_rate': 'Learning rate',
            'epochs': 'Epochs',
            'eval_dataset': 'Evaluation dataset',
        }

        if 'fft' not in args.strategy:
            required_params.update({
                'rank': 'LoRA rank',
                'target_modules': 'LoRA target modules'
            })
        
        missing_params = []
        for param, description in required_params.items():
            if getattr(args, param) is None:
                missing_params.append(f'--{param} ({description})')
        
        if missing_params:
            raise ValueError(f"The following required parameters are missing: {', '.join(missing_params)}")
        
        if args.sweep_method == 'bayes':
            if len(args.learning_rate) < 2:
                raise ValueError("When using bayes method, learning rate needs to specify maximum and minimum values, and will be optimized within the range")
            if args.sweep_count < 1:
                raise ValueError("Sweep count must be greater than 0")
            
        # Validate numeric parameters must be greater than 0
        numeric_params = [
            ('epochs', args.epochs, 'Epochs'),
            ('rank', args.rank, 'Rank'),
            ('batch_size', args.batch_size, 'Batch size'),
            ('seed', args.seed, 'Random seed')
        ]
        
        for _, param_value, param_desc in numeric_params:
            if isinstance(param_value, list) and any(val < 1 for val in param_value) or \
                isinstance(param_value, int) and param_value < 1:
                raise ValueError(f"{param_desc} must be greater than 0 and must be specified as integer")
        
        float_params = [
            ('learning_rate', args.learning_rate, 'Learning rate'),
            ('warmup_ratio', args.warmup_ratio, 'Warmup ratio')
        ]
        
        for _, param_value, param_desc in float_params:
            if isinstance(param_value, list) and any(val <= 0 for val in param_value) or \
                isinstance(param_value, float) and param_value <= 0:
                raise ValueError(f"{param_desc} must be positive")
        
        if args.target_modules and \
            any(target_module not in ['q_proj', 'k_proj', 'v_proj', 'o_proj', \
                                      'qkv_proj', 'gate_proj', 'down_proj', 'up_proj'] \
                                        for target_module in args.target_modules):
            raise ValueError("Target modules must be specified as q_proj, k_proj, v_proj, o_proj, qkv_proj, gate_proj, down_proj or up_proj")
        
        # if args.model_name and 'phi' in args.model_name.lower() and args.target_modules:
        #     if any(module in ['q_proj', 'k_proj', 'v_proj'] for module in args.target_modules):
        #         raise ValueError("For Microsoft's phi series models, use qkv_proj instead of q_proj, k_proj, v_proj")
    
    return args 