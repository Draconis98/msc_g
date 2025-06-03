"""Module for managing model evaluation using OpenCompass framework."""

import math
import os
import csv
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
        self.summary_dir = None
        
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
            seq_memory = MAX_SEQ_LEN * bytes_per_token * 3

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

    def _get_latest_eval_dir(self):
        """Get the directory of the latest evaluation run."""
        try:
            eval_runs = [
                d for d in os.listdir(self.eval_dir)
                if os.path.isdir(os.path.join(self.eval_dir, d))
            ]
            if not eval_runs:
                raise RuntimeError("No evaluation runs found")
                
            latest_run = max(
                eval_runs,
                key=lambda x: os.path.getctime(os.path.join(self.eval_dir, x))
            )
            return os.path.join(self.eval_dir, latest_run, "summary")
        except Exception as e:
            logger.error("Error getting latest run directory: %s", str(e))
            raise

    def _process_csv_file(self, filepath):
        """Process a single CSV file and log results to wandb."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Get the last column name (model name)
                fieldnames = reader.fieldnames
                if not fieldnames:
                    raise ValueError(f"No fields found in CSV file: {filepath}")
                    
                model_column = fieldnames[-1]
                processed_rows = 0
                
                for row in reader:
                    try:
                        self._log_row_to_wandb(row, model_column)
                        processed_rows += 1
                    except (KeyError, ValueError) as e:
                        logger.warning("Error processing row in %s: %s" % (filepath, str(e)))
                        
                logger.info("Processed %d rows from %s" % (processed_rows, filepath))
                
        except Exception as e:
            logger.error("Error processing file %s: %s" % (filepath, str(e)))
            raise
            
    def _log_row_to_wandb(self, row, model_column):
        """Log a single row of results to wandb."""
        try:
            dataset = row['dataset']
            version = row['version']
            metric = row['metric']
            mode = row['mode']
            value = float(row[model_column])
            
            metric_name = f"eval/{dataset}-{mode}({metric})"
            wandb.log({metric_name: value})
            
        except KeyError as e:
            raise KeyError(f"Missing required column: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Invalid value in {model_column} column: {str(e)}") from e
            
    def _process_all_results(self):
        """Process all CSV files in the summary directory."""
        logger.info(f"Processing results from {self.summary_dir}")
        
        try:
            csv_files = [
                f for f in os.listdir(self.summary_dir)
                if f.endswith('.csv')
            ]
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise
            
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {self.summary_dir}")

        for filename in csv_files:
            filepath = os.path.join(self.summary_dir, filename)
            self._process_csv_file(filepath)
                
        logger.success("All results processed successfully")

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
        self.summary_dir = self._get_latest_eval_dir()
        self._process_all_results()