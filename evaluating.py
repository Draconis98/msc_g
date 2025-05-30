"""Module for managing model evaluation using OpenCompass framework."""

import math
import os
import subprocess

import GPUtil
import torch
import wandb
from loguru import logger
from peft import PeftConfig
from transformers import AutoModelForCausalLM

from utils.config import OPENCOMPASS_DIR, OUTPUT_DIR

MAX_SEQ_LEN = 512

class LLMEvaluatingPipeline:
    """Manages the complete evaluation process including setup, running, and results processing."""
    
    def __init__(self, config):
        """Initialize evaluator with configuration and output directory."""
        self.config = config
        self.output_dir = os.path.join(OUTPUT_DIR, wandb.run.name.replace("-", "/"))
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.model_config = None
        self.config_filename = None
        
    def _find_optimal_batch_size(self):
        """Calculate batch size based on available GPU memory and model size."""
        try:
            # Load model to get its memory footprint
            peft_config = PeftConfig.from_pretrained(self.output_dir)
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.bfloat16 if self.config['bf16'] else torch.float32,
                trust_remote_code=True
            ).to('cuda')
            
            # Get available GPUs
            gpus = GPUtil.getGPUs()
            
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                gpus = [gpu for gpu in gpus if gpu.id in list(map(int, cuda_visible_devices.split(',')))]
            
            if not gpus:
                raise RuntimeError("No GPU available")
                
            available_memory = gpus[0].memoryTotal - gpus[0].memoryUsed
            del model
            torch.cuda.empty_cache()
            
            # Estimate memory per sequence
            bytes_per_token = 2  # bfloat16
            seq_memory = MAX_SEQ_LEN * bytes_per_token * 2

            batch_size = math.ceil(available_memory / seq_memory)
                
            batch_size = int(max(1, batch_size - (3 if batch_size % 2 == 0 else 1)))
            logger.success(f"Calculated batch size: {batch_size} (Available memory: {available_memory}MB)")
            return batch_size
            
        except (RuntimeError, OSError, torch.cuda.CudaError) as e:
            logger.warning(f"Failed to determine optimal batch size: {str(e)}")
            return 32  # fallback to default
        
    def _get_model_config(self):
        """Generate model configuration for OpenCompass."""
        # Determine optimal batch size
        batch_size = self._find_optimal_batch_size()
        peft_config = PeftConfig.from_pretrained(self.output_dir)
        
        try:
            return {
                'abbr': f"{self.config['model_name'].split('/')[1]}-{self.config['strategy']}",
                'type': 'HuggingFacewithChatTemplate',
                'path': peft_config.base_model_name_or_path,  # Use base model path
                'model_kwargs': {
                    'torch_dtype': 'torch.bfloat16' if self.config['bf16'] else 'torch.float32',
                    'trust_remote_code': True,
                },
                'tokenizer_kwargs': {
                    'padding_side': 'left',
                    'truncation_side': 'left',
                    'trust_remote_code': True
                },
                'peft_path': self.output_dir,  # Add LoRA adapter path
                'max_out_len': MAX_SEQ_LEN,
                'batch_size': batch_size,
                'run_cfg': {'num_gpus': 1, 'num_procs': 1}, # TODO: change to multi-gpu
            }
        except Exception as e:
            logger.error("Failed to get model configuration: %s", str(e))
            raise

    def opencompass_configuration(self):
        """Setup opencompass configuration for evaluation."""
        logger.info("Setting up opencompass configuration...")
        
        # Create model directory
        try:
            model_name = self.config['model_name'].split('/')[1].split('-')[0].lower()
            eval_dir = os.path.join(OPENCOMPASS_DIR, 'opencompass', 'configs', 'models', model_name)
            os.makedirs(eval_dir, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create model directory in opencompass: %s", str(e))
            raise

        # Create model configuration file
        try:
            self.model_config = self._get_model_config()
            self.config_filename = f"{self.config['model_name'].split('/')[1]}.py"
            file_path = os.path.join(eval_dir, self.config_filename)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('from opencompass.models import HuggingFacewithChatTemplate\n')
                file.write('import torch\n\n')
                file.write('models = [\n')
                file.write('    dict(\n')
                for key, value in self.model_config.items():
                    if isinstance(value, dict):
                        file.write(f'        {key}=dict(\n')
                        for k, v in value.items():
                            if isinstance(v, str):
                                file.write(f'            {k}="{v}",\n')
                            else:
                                file.write(f'            {k}={v},\n')
                        file.write('        ),\n')
                    else:
                        if isinstance(value, str):
                            file.write(f"        {key}='{value}',\n")
                        else:
                            file.write(f"        {key}={value},\n")
                file.write('    ),\n')
                file.write(']\n')
        except Exception as e:
            logger.error("Failed to create model configuration file: %s", str(e))
            raise

        # Log evaluation configuration to wandb
        wandb.log({
            "evaluation_config": {
                "model_name": self.config['model_name'].split('/')[1],
                "strategy": self.config['strategy'],
                "config_path": file_path,
                "model_config": self.model_config
            }
        })
        
        logger.success("Opencompass configuration completed")

    def evaluation(self):
        """Run the evaluation process."""
        logger.info("Running evaluation...")
        opencompass_run = os.path.join(OPENCOMPASS_DIR, "run.py")
        
        try:
            result = subprocess.run([
                "python", opencompass_run,
                "--models", self.config_filename,
                *["--datasets"] + [dataset for dataset in self.config['eval_dataset']],
                "-w", self.eval_dir
            ], check=True, capture_output=True, text=True, encoding="utf-8")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("Evaluation failed: %s", e.stderr)
            raise
            
        logger.success("Evaluation completed")
    
    def run(self):
        """Run the complete evaluation pipeline."""
        self.opencompass_configuration()
        self.evaluation()