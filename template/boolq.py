boolq_reader = dict(
    input_columns=['question', 'passage'],
    output_column='label',
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"{example['passage']}\nQuestion: {example['question']}\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": "A. Yes" if (isinstance(example["label"], bool) and example["label"]) or (isinstance(example["label"], int) and example["label"] == 0) else "B. No"}
    ]
    return messages