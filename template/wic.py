wic_reader = dict(
    input_columns=[
        'word',
        'sentence1',
        'sentence2',
    ],
    output_column='label',
)

def get_datasets(example):
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    word = example['word']
    label = example['label']
    messages = [
        {"role": "user", "content": f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre '{word}' in the above two sentenses the same?\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": "A. Yes" if label else "B. No"}
    ]
    return messages