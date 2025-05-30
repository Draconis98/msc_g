# Codes

A Python project for efficient LLM fine-tuning and evaluation, integrating OpenCompass and PEFT.

## Features
* Support parameter-efficient fine-tuning (PEFT/LoRA/FFT etc.) for various large models
* One-stop training and evaluation pipeline with automated configuration and logging
* Integrate OpenCompass as evaluation benchmark
* Support batch experiments with multiple datasets, models and strategies
* Full GPU acceleration for training and evaluation, with WandB logging support

## Directory Structure

```
.
├── main.py                # Main entry point, command line argument parsing and dispatching
├── pipeline.py            # Training and evaluation pipeline
├── training.py            # Training process implementation
├── evaluating.py          # Evaluation process implementation, integrating OpenCompass
├── data_processor.py      # Data loading and preprocessing
├── parse.py               # Command line argument parsing
├── utils/                 # Configuration and utility functions
├── opencompass/           # OpenCompass submodule (evaluation benchmark)
├── peft/                  # Huggingface PEFT submodule (parameter-efficient fine-tuning)
├── outputs/               # Training and evaluation outputs
└── ...
```

## Submodules

- **opencompass**  
  An open-source framework for large model evaluation, supporting various mainstream benchmarks and datasets. See [OpenCompass Documentation](https://github.com/open-compass/opencompass) for details.

- **peft**  
  Huggingface's official parameter-efficient fine-tuning library, supporting methods like LoRA, Prompt Tuning, etc. See [PEFT Documentation](https://github.com/huggingface/peft) for details.

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:Draconis98/msc_g.git
cd msc_g
```

### 2. Python Environment
#### Environment Management Tool [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv init
uv venv .venv --python python3.10
cd opencompass
uv pip install -e .
cd .. && cd peft
uv pip install -e .
uv pip install logulu wandb gputil
```

## Usage

### Integrated Training and Evaluation Command

```bash
python main.py \
  -s fft,lora \
  -m Qwen/Qwen3-0.6B,Qwen/Qwen2-0.5B \
  -d Idavidrein/gpqa \
  -lr 1e-4,1e-5 \
  -ed gpqa_gen \
  -e 1,2 \
  -tt CAUSAL_LM \
  -r 16,32 \
  -t q_proj,k_proj,v_proj
```

**Key Parameters:**
- `-s, --strategy`: Training strategy (e.g., fft, lora, pissa, etc., multiple choices allowed)
- `-m, --model_name`: Model name (in huggingface format, multiple choices allowed)  
- `-d, --dataset`: Training dataset
- `-lr, --learning_rate`: Learning rate (multiple choices allowed)
- `-ed, --eval_dataset`: Evaluation dataset
- `-e, --epochs`: Number of training epochs
- `-tt, --task_type`: Task type (e.g., CAUSAL_LM, etc.)
- `-r, --rank`: LoRA rank
- `-t, --target_modules`: LoRA target modules

For more parameters, see `parse.py` and `utils/config.py`, which support customizing batch size, gradient accumulation, max sequence length, etc.

### Outputs and logs

- Training and evaluation results are saved in the `outputs/` directory
- WandB is used for experiment tracking and visualization

## OpenCompass

This project automatically generates OpenCompass configurations and invokes its evaluation pipeline. You can also enter the `opencompass/` directory separately and refer to its [official documentation](https://github.com/open-compass/opencompass) for custom evaluation.

## Reference

- [OpenCompass](https://github.com/open-compass/opencompass): Large Language Model Evaluation Benchmark
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning Library