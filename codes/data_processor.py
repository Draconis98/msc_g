from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from transformers import AutoTokenizer
import logging
import os
from utils.config import MODEL_PATHS, DATASETS_DIR

from template.data_mapping import dataset_mapping

def get_tokenizer(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return tokenizer

def get_model_checkpoint(model_name):
    name, size = model_name.split(':')
    return MODEL_PATHS[name][size]

def apply_chat_template(dataset_name, example, tokenizer):
    # Get the template module path from dataset_mapping
    _, _, template_path = dataset_mapping[dataset_name]
    
    # Dynamically import the correct template module
    template_module = __import__(template_path, fromlist=['get_datasets'])
    messages = template_module.get_datasets(example)
    
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

def process_chat_template(dataset_name, datasets, tokenizer, columns_names, batched):
    return datasets.map(
        lambda x: apply_chat_template(dataset_name, x, tokenizer),
        batched=batched,
        num_proc=100,
        remove_columns=columns_names,
        desc="Applying chat template"
    )

def load_and_process_data(config):
    # Get tokenizer
    checkpoint_path = get_model_checkpoint(config['model_name'])
    tokenizer = get_tokenizer(checkpoint_path)
    
    # Load dataset
    dataset_name, dataset_split, _ = dataset_mapping[config['dataset']]
    all_configs = get_dataset_config_names(dataset_name)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    if len(all_configs) == 0:
        dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True, cache_dir=DATASETS_DIR)
    else:
        datasets_list = [load_dataset(dataset_name, config_name, split=dataset_split, trust_remote_code=True, cache_dir=DATASETS_DIR)
                        for config_name in all_configs]
        dataset = concatenate_datasets(datasets_list)
    
    # Process dataset
    columns_names = list(dataset.column_names)
    batched = config['dataset_batched']
    processed_dataset = process_chat_template(config['dataset'], dataset, tokenizer, columns_names, batched)
    logging.info(processed_dataset[0])
    
    return tokenizer, processed_dataset 