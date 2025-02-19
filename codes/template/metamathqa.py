reader = dict(
    input_columns=['query'],
    output_column='response'
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"Question: {example['query']}\nLet's think step by step\nAnswer:"},
        {"role": "assistant", "content": example["response"]}
    ]
    resp = example["response"]
    return messages, resp