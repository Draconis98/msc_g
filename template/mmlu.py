reader = dict(
    input_columns=['question', 'choices'],
    output_column='answer',
    train_split='auxiliary_train'
)

mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

def get_datasets(example):
    _hint = f'Answer the question by replying A, B, C or D.'
    question = example['question']
    choices = example['choices']
    answer = example['answer']
    messages = [
        {"role": "user", "content": f"{_hint}\nQuestion: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: "},
        {"role": "assistant", "content": "A" if answer == 0 else "B" if answer == 1 else "C" if answer == 2 else "D"}
    ]

    return messages, str(answer)