"""Module for loading and processing datasets with chat templates."""

import os
import logging
import torch
import numpy as np

from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from transformers import AutoTokenizer

from utils.config import MODEL_PATHS, DATASETS_DIR
from template.data_mapping import dataset_mapping


def get_tokenizer(checkpoint_path):
    """Get a tokenizer for a given checkpoint path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        return tokenizer
    except Exception as e:
        logging.error("Failed to get tokenizer: %s", str(e))
        raise

def get_model_checkpoint(model_name):
    """Get a model checkpoint for a given model name."""
    try:
        name, size = model_name.split(':')
        return MODEL_PATHS[name][size]
    except Exception as e:
        logging.error("Failed to get model checkpoint: %s", str(e))
        raise

def apply_chat_template(dataset_name, example, tokenizer):
    """Apply a chat template to an example dataset."""
    # Get the template module path from dataset_mapping
    _, _, _, template_path = dataset_mapping[dataset_name]
    
    # Dynamically import the correct template module
    template_module = __import__(template_path, fromlist=['get_datasets'])
    messages, resp = template_module.get_datasets(example)
    
    # Create a new dictionary with only text and resp
    result = {
        "text": tokenizer.apply_chat_template(messages, tokenize=False),
        "resp": tokenizer.encode(resp, add_special_tokens=False)
    }
    return result

def process_chat_template(dataset_name, datasets, tokenizer, columns_names, batched):
    """Process a dataset with a chat template."""
    return datasets.map(
        lambda x: apply_chat_template(dataset_name, x, tokenizer),
        batched=batched,
        num_proc=100,
        remove_columns=columns_names,
        desc="Applying chat template",
        load_from_cache_file=False  # disable cache loading
    )

def load_and_process_data(config):
    """Load and process data for a given configuration."""
    # Get tokenizer
    checkpoint_path = get_model_checkpoint(config['model_name'])
    tokenizer = get_tokenizer(checkpoint_path)
    
    # Load dataset
    dataset_name, config_name, dataset_split, _ = dataset_mapping[config['dataset']]
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Get required columns from template
    # template_module = __import__(template_path, fromlist=['reader'])
    # required_columns = template_module.reader['input_columns']
    
    # Load dataset with config_name if specified
    if config_name is not None:
        if ',' in config_name:
            # Handle multiple config names
            config_names = config_name.split(',')
            datasets_list = [
                load_dataset(
                    dataset_name,
                    config,
                    split=dataset_split,
                    trust_remote_code=True,
                    cache_dir=DATASETS_DIR,
                    # select_columns=required_columns
                )
                for config in config_names
            ]
            dataset = concatenate_datasets(datasets_list)
        else:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=dataset_split,
                trust_remote_code=True,
                cache_dir=DATASETS_DIR,
                # select_columns=required_columns
            )
    else:
        dataset = load_dataset(
            dataset_name,
            split=dataset_split,
            trust_remote_code=True,
            cache_dir=DATASETS_DIR,
            # select_columns=required_columns
        )
        
    
    # Process dataset
    columns_names = list(dataset.column_names)
    batched = config['dataset_batched']
    processed_dataset = process_chat_template(config['dataset'], dataset, tokenizer, columns_names, batched)

    logging.info("Processed dataset case: %s", processed_dataset[0])

    if config["data_selection"]:
        logging.info("Calculating entropy")
        processed_dataset = calculate_metric(processed_dataset, 'entropy')
        logging.info("Calculating perplexity") 
        processed_dataset = calculate_metric(processed_dataset, 'perplexity')
        
        entropy_mean = np.mean(processed_dataset['entropy'])
        perplexity_mean = np.mean(processed_dataset['perplexity'])

        processed_dataset = processed_dataset.filter(lambda x: x['entropy'] > entropy_mean and x['perplexity'] > perplexity_mean)
        
        logging.info("Filtered dataset size: %s", len(processed_dataset))
    
    return tokenizer, processed_dataset 


def calculate_metric(dataset, metric_type):
    """Calculate the entropy or perplexity of each data in a dataset using GPU if available.
    
    Args:
        dataset: The input dataset
        metric_type: Either 'entropy' or 'perplexity'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Move the entire dataset to GPU
    token_ids_tensors = []
    for data in dataset:
        # resp is already encoded, just convert to tensor
        token_ids_tensors.append(torch.tensor(data['resp'], device=device))
    
    scores = []
    
    try:
        # Process all tensors on GPU
        for token_ids_tensor in token_ids_tensors:
            # Calculate frequency of each token on GPU
            _, counts = torch.unique(token_ids_tensor, return_counts=True)
            counts = counts.to(device)
            probs = counts.float() / len(token_ids_tensor)
            
            if metric_type == 'perplexity':
                # Calculate perplexity directly using probabilities
                text_perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
                score = text_perplexity
            else:
                # Calculate entropy using Shannon's formula
                text_entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                score = text_entropy
                
            scores.append(score.cpu().item())  # Move result back to CPU for storage
            
            # Clean up GPU memory
            del counts, probs
            torch.cuda.empty_cache()
            
        # Clean up token tensors
        for tensor in token_ids_tensors:
            del tensor
        torch.cuda.empty_cache()
            
    except Exception as e:
        # Clean up on error
        for tensor in token_ids_tensors:
            del tensor
        torch.cuda.empty_cache()
        raise e
    
    # Add scores to dataset
    dataset = dataset.add_column(metric_type, scores)
    
    return dataset