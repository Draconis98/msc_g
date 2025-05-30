"""Module for loading and processing datasets with chat templates."""

from loguru import logger
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from template.data_mapping import dataset_mapping


def get_tokenizer(model_name):
    """Get a tokenizer for a given checkpoint path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to get tokenizer: {str(e)}")
        raise

def apply_chat_template(dataset_name, example, tokenizer):
    """Apply a chat template to an example dataset."""
    # Get the template module path from dataset_mapping
    _, _, template_path = dataset_mapping[dataset_name]
    
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

def load_and_process_data(model_name, dataset_name, batched):
    """Load and process data for a given configuration."""
    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Load dataset
    subset, split, _ = dataset_mapping[dataset_name]
    
    # Load dataset with subset if specified
    if subset is None:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    elif ',' not in subset:
        dataset = load_dataset(dataset_name, name=subset, split=split, trust_remote_code=True)
    else:
        subsets = subset.split(',')
        datasets_list = [
            load_dataset(dataset_name, name=subset, split=split, trust_remote_code=True)
            for subset in tqdm(subsets, desc="Loading datasets")
        ]
        dataset = concatenate_datasets(datasets_list)
    
    # Process dataset
    columns_names = list(dataset.column_names)
    
    processed_dataset = process_chat_template(dataset_name, dataset, tokenizer, columns_names, batched)

    return tokenizer, processed_dataset