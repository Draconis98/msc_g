metamathqa_reader = dict(
    input_columns=['question', 'choices'],
    output_column='answerKey',
    train_split='train'
)

def get_datasets(example):
    messages = [
        {"role": "user", "content": f"{example['question']}\nA. {example['choices']['text'][0]}\nB. {example['choices']['text'][1]}\nC. {example['choices']['text'][2]}\nD. {example['choices']['text'][3]}\nE. {example['choices']['text'][4]}\nAnswer:"},
        {"role": "assistant", "content": example["answerKey"]}
    ]
    resp = example["answerKey"]
    return messages, resp