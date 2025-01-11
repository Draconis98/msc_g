dataset_mapping = {
        'metamath': ("meta-math/MetaMathQA", "train[:100000]", "template.metamathqa"),
        'mmlu': ("lukaemon/mmlu", "train", "template.mmlu"),
        'super_glue/axb': ("aps/super_glue", "train", "template.axb"),
        'super_glue/boolq': ("aps/super_glue", "train", "template.boolq"),
        'super_glue/cb': ("aps/super_glue", "train", "template.cb"),
        'super_glue/copa': ("aps/super_glue", "train", "template.copa"),
        'super_glue/multirc': ("aps/super_glue", "train", "template.multirc"),
        'super_glue/record': ("aps/super_glue", "train", "template.record"),
        'super_glue/rte': ("aps/super_glue", "train", "template.rte"),
        'super_glue/wsc': ("aps/super_glue", "train", "template.wsc"),
        'super_glue/wic': ("aps/super_glue", "train", "template.wic"),
        'commonsense_qa': ("tau/commonsense_qa", "train", "template.commonsense_qa"),
        # Add other datasets here as needed
        }