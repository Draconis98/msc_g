rte_reader = dict(
    input_columns=['hypothesis', 'premise'],
    output_column='label',
)

def get_datasets(example):
    premise = example['premise']
    hypothesis = example['hypothesis']
    label = example['label']
    messages = [
        {"role": "user", "content": f"{premise}\n{hypothesis}\nIs the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": "A. Yes" if label == "entailment" else "B. No"}
    ]
    return messages