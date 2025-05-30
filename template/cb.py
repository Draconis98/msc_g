cb_reader = dict(
    input_columns=['premise', 'hypothesis'],
    output_column='label',
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"{example['premise']}\n{example['hypothesis']}\nWhat is the relation between the two sentences?\nA. Contradiction\nB. Entailment\nC. Neutral\nAnswer:"},
        {"role": "assistant", "content": "A. Contradiction" if example["label"] == "contradiction" else "B. Entailment" if example["label"] == "entailment" else "C. Neutral"}
    ]
    return messages