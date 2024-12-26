metamathqa_reader = dict(
    input_columns=['query'],
    output_column='response',
    train_split='train[:100000]'
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"Question: {example['query']}\nLet's think step by step\nAnswer:"},
        {"role": "assistant", "content": example["response"]}
    ]
    return messages