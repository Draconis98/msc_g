import json


record_reader = dict(
    input_columns=['passage', 'query'],
    output_column='answers',
)

def get_datasets(example):
    passage = example['passage']
    passage = passage.replace('@highlight', '')
    query = example['query']
    query = query.replace('@placeholder', '____')
    answers = example['answers']
    answers_str = ", ".join(answers) if isinstance(answers, list) else answers
    messages = [
        {"role": "user", "content": f"Passage: {passage}\nResult: {query}\nQuestion: What entity does ____ refer to in the result? Give me the entity name:"},
        {"role": "assistant", "content": answers_str}
    ]
    return messages