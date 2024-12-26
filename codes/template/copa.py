copa_reader = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='label',
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"{example['premise']}\nQuestion: Which may be the {example['question']}?\nA. {example['choice1']}\nB. {example['choice2']}\nAnswer:"},
        {"role": "assistant", "content": "A" if example["label"] == 0 else "B"}
    ]
    return messages