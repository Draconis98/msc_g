reader = dict(
    input_columns=['question'],
    output_column='r1_solution_1'
)

def get_datasets(example):
    ans = example["r1_solution_1"].split("</think>")[-1]
    messages = [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": ans}
    ]
    return messages, ans