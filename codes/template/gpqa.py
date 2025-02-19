import random

reader = dict(
    input_columns=['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Explanation'],
    output_column='answer')

def get_datasets(example):
    """Convert example to messages format for multiple choice QA."""
    question = example['Question']
    correct_answer = example['Correct Answer']
    incorrect_answer1 = example['Incorrect Answer 1']
    incorrect_answer2 = example['Incorrect Answer 2']
    incorrect_answer3 = example['Incorrect Answer 3']
    explanation = example['Explanation']
    
    options = [correct_answer, incorrect_answer1, incorrect_answer2, incorrect_answer3]
    random.shuffle(options)
    A, B, C, D = options
    correct_letter = ['A', 'B', 'C', 'D'][options.index(correct_answer)]
    
    messages = [
        {"role": "user", "content": f"Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n{question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:"},
        {"role": "assistant", "content": f"{explanation}\nANSWER: {correct_letter}"}
    ]
    resp = f"{explanation}\nANSWER: {correct_letter}"
    return messages, resp