multirc_reader = dict(
    input_columns=['question', 'paragraph', 'answer'],
    output_column='label',
)

def get_datasets(example):
    paragraph = example['paragraph']
    question = example['question']
    answer = example['answer']
    label = example['label']
    messages = [
        {"role": "user", "content": f"{paragraph}\nQuestion: {question}\nAnswer: {answer}\nIs it true?\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": "A. Yes" if label else "B. No"}
    ]
    return messages