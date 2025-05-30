wsc_reader = dict(
    input_columns=['span1_text', 'span2_text', 'text'],
    output_column='label',
)

def get_datasets(example):
    # Validate input data
    required_fields = ['text', 'span1_text', 'span2_text', 'label']
    for field in required_fields:
        if field not in example or example[field] is None:
            raise ValueError(f"Missing or None value for required field: {field}")

    text = example['text']
    span1 = example['span1_text']
    span2 = example['span2_text']
    label = example['label']
    
    # Convert label to boolean if it's not already
    if isinstance(label, str):
        label = label.lower() == 'true'
    elif not isinstance(label, bool):
        raise ValueError(f"Invalid label type: {type(label)}. Expected bool or str.")
    
    messages = [
        {"role": "user", "content": f"Passage: {text}\nDoes the pronoun # {span2} # refer to * {span1} *?\nA. Yes\nB. No\nAnswer:"},
        {"role": "assistant", "content": "A. Yes" if label else "B. No"}
    ]
    return messages
