dataset_mapping = {
        'metamath': ("meta-math/MetaMathQA", "train[:100000]", "template.metamathqa"),
        'mmlu': ("lukaemon/mmlu", "train", "template.mmlu"),
        'super_glue/axb': ("aps/super_glue", "train", "template.axb"),
        'super_glue/boolq': ("aps/super_glue", "train", "template.boolq"),
        'super_glue/cb': ("aps/super_glue", "train", "template.cb"),
        'super_glue/copa': ("aps/super_glue", "train", "template.copa"),
        'super_glue/record': ("aps/super_glue", "train", "template.record"),
        'commonsense_qa': ("tau/commonsense_qa", "train", "template.commonsense_qa"),
        # Add other datasets here as needed
        }