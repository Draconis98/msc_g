"""Module for managing model evaluation using OpenCompass framework."""

import math
import os
from loguru import logger
import subprocess

import torch
import wandb
from transformers import AutoModelForCausalLM
import GPUtil

from utils.config import OPENCOMPASS_DIR
from utils.misc import get_output_dir

MAX_SEQ_LEN = 4096

class EvaluatingPipeline:
    """Manages the complete evaluation process including setup, running, and results processing."""
    
    def __init__(self, config):
        """Initialize evaluator with configuration and output directory."""
        self.config = config
        self.output_dir = get_output_dir(config)
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.model_config = None
        self.config_filename = None
        
    def _find_optimal_batch_size(self):
        """Calculate batch size based on available GPU memory and model size."""
        try:
            # Load model to get its memory footprint
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                torch_dtype=torch.bfloat16,
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
            seq_memory = MAX_SEQ_LEN * bytes_per_token

            batch_size = math.ceil(available_memory / seq_memory)
                
            batch_size = int(max(1, batch_size - (3 if batch_size % 2 == 0 else 1)))
            logger.info("Calculated batch size: %d (Available memory: %.2fMB)", batch_size, available_memory)
            return batch_size
            
        except (RuntimeError, OSError, torch.cuda.CudaError) as e:
            logger.warning("Failed to determine optimal batch size: %s", str(e))
            return 32  # fallback to default
        
    def _get_model_config(self):
        """Generate model configuration for OpenCompass."""
        # Determine optimal batch size
        batch_size = self._find_optimal_batch_size()
        
        try:
            return {
                'abbr': f"{self.config['model_name'].replace(':', '-')}-{self.config['strategy']}",
                'type': 'HuggingFacewithChatTemplate',
                'path': self.output_dir,
                'model_kwargs': {
                    'torch_dtype': 'torch.bfloat16',
                },
                'tokenizer_kwargs': {
                    'padding_side': 'left',
                    'truncation_side': 'left',
                    'trust_remote_code': True,
                },
                'max_out_len': MAX_SEQ_LEN,
                'max_seq_len': MAX_SEQ_LEN,
                'batch_size': batch_size,
                'run_cfg': {'num_gpus': 1, 'num_procs': 1},
            }
        except Exception as e:
            logger.error("Failed to get model configuration: %s", str(e))
            raise

    def opencompass_configuration(self):
        """Setup opencompass configuration for evaluation."""
        logger.info("Setting up opencompass configuration...")
        
        # Create model directory
        try:
            model_name = self.config['model_name'].split(':')[0]
            model_name = ''.join([i for i in model_name if not i.isdigit() and i != '.'])
            eval_dir = os.path.join(OPENCOMPASS_DIR, 'opencompass', 'configs', 'models', model_name)
            os.makedirs(eval_dir, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create model directory in opencompass: %s", str(e))
            raise

        # Create model configuration file
        try:
            self.model_config = self._get_model_config()
            self.config_filename = f"{self.config['model_name'].replace(':', '-')}.py"
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
                "model_name": self.config['model_name'],
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
                *["--datasets"] + [dataset + "_gen" for dataset in self.config['eval_dataset']],
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