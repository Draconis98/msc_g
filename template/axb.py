axb_reader = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"{example['sentence1']}\n{example['sentence2']}\nIs the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": example["label"]}
    ]
    return messages