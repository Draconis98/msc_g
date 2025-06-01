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

- **OpenCompass**  
  An open-source framework for large model evaluation, supporting various mainstream benchmarks and datasets. See [<img src="https://img.shields.io/badge/GitHub-OpenCompass-blue?logo=github" align="center">](https://github.com/open-compass/opencompass) for details.

- **PEFT**  
  Huggingface's official parameter-efficient fine-tuning library, supporting methods like LoRA, Prompt Tuning, etc. See [<img src="https://img.shields.io/badge/🤗-PEFT-blue" align="center">](https://huggingface.co/docs/peft/index) for details.

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
source .venv/bin/activate
cd opencompass
uv pip install -e .
cd .. && cd peft
uv pip install -e .
uv pip install trl logulu wandb gputil ninja
uv pip install flash-attn --no-build-isolation
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

This project automatically generates OpenCompass configurations and invokes its evaluation pipeline. You can also enter the `opencompass/` directory separately and refer to its [<img src="https://img.shields.io/badge/GitHub-OpenCompass-blue?logo=github" align="center">](https://github.com/open-compass/opencompass) for custom evaluation.

## Reference

- [<img src="https://img.shields.io/badge/🤗-HuggingFace-blue" align="center">](https://huggingface.co) is a company and open-source ecosystem that has become a central hub for natural language processing (NLP) and AI development. It’s widely used by researchers, developers, and companies to build, share, and deploy machine learning models—especially Transformer-based models like BERT, GPT, T5, etc.
- [<img src="https://img.shields.io/badge/GitHub-OpenCompass-blue?logo=github" align="center">](https://github.com/open-compass/opencompass) is a Large Language Model Evaluation Benchmark
- [<img src="https://img.shields.io/badge/GitHub-Transformers-blue?logo=github" align="center">](https://github.com/huggingface/transformers) [<img src="https://img.shields.io/badge/🤗-Transformers-blue" align="center">](https://huggingface.co/docs/transformers/index) is a library of pretrained natural language processing, computer vision, audio, and multimodal models for inference and training. 
- [<img src="https://img.shields.io/badge/GitHub-PEFT-blue?logo=github" align="center">](https://github.com/huggingface/peft) [<img src="https://img.shields.io/badge/🤗-PEFT-blue" align="center">](https://huggingface.co/docs/peft/index) is a library for efficiently adapting large pretrained models to various downstream applications without fine-tuning all of a model’s parameters because it is prohibitively costly.
- [<img src="https://img.shields.io/badge/GitHub-TRL-blue?logo=github" align="center">](https://github.com/huggingface/trl) [<img src="https://img.shields.io/badge/🤗-TRL-blue" align="center">](https://huggingface.co/docs/trl/index) is a full stack library where we provide a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more.
- [<img src="https://img.shields.io/badge/GitHub-FlashAttention-blue?logo=github" align="center">](https://github.com/Dao-AILab/flash-attention) is an optimized algorithm for computing attention in Transformer models, designed to be faster and more memory-efficient than standard implementations. [<img src="https://img.shields.io/badge/📄-FlashAttention1-green" align="center">](https://arxiv.org/abs/2205.14135) [<img src="https://img.shields.io/badge/📄-FlashAttention2-green" align="center">](https://tridao.me/publications/flash2/flash2.pdf)