dataset_mapping = {
        "meta-math/MetaMathQA": (None, "train[:100000]", "template.metamathqa"),
        "open-r1/OpenR1-Math-220k": ("default", "train", "template.openr1math"),
        "cais/mmlu": ("all", "auxiliary_train", "template.mmlu"),
        "aps/super_glue": [
            (name, "train", f"template.{name}") for name in [
                "axb", "boolq", "cb", "copa", "multirc", "record", "rte", "wsc", "wic"
            ]
        ],
        "tau/commonsense_qa": (None, "train", "template.commonsense_qa"),
        "Idavidrein/gpqa": ("gpqa_main,gpqa_extended", "train", "template.gpqa"),
        "TsinghuaC3I/MedXpertQA": (None, "train", "template.medxpertqa"),
        "EricLu/SCP-116K": (None, "train", "template.scp"),
        "zwhe99/DeepMath-103K": (None, "train", "template.deepmath"),
        }
