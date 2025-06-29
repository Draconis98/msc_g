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

class LLMEvaluatingPipeline:
    """Manages the complete evaluation process including setup, running, and results processing."""
    
    def __init__(self, config, resume=False):
        """Initialize evaluator with configuration and output directory."""
        self.config = config
        self.output_dir = None
        self.model_config = None
        self.config_filename = None
        self.summary_dir = None
        self.batch_size = None
        self.resume = resume

        if not self.resume:
            self.output_dir = os.path.join(OUTPUT_DIR, wandb.run.name.replace("-", "/"))
        else:
            self.output_dir = self.config['output_dir']
        self.eval_dir = os.path.join(self.output_dir, "eval")

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
            seq_memory = self.config['max_out_len'] * bytes_per_token * 2

            batch_size = math.ceil(available_memory / seq_memory)
                
            batch_size = int(max(1, batch_size))
            logger.success(f"Calculated batch size: {batch_size} (Available memory: {available_memory}MB)")
            self.batch_size = batch_size
            
        except (RuntimeError, OSError, torch.cuda.CudaError) as e:
            logger.warning(f"Failed to determine optimal batch size: {str(e)}")
            self.batch_size = 32  # fallback to default
        
    def _get_model_config(self):
        """Generate model configuration for OpenCompass."""
        # Determine optimal batch size
        if not self.batch_size:
            self._find_optimal_batch_size()
        peft_config = PeftConfig.from_pretrained(self.output_dir)
        
        try:
            return dict(
                abbr=f"{self.config['model_name'].split('/')[1]}-{self.config['strategy']}",
                type="HuggingFacewithChatTemplate",
                path=peft_config.base_model_name_or_path,
                model_kwargs=dict(
                    torch_dtype="torch.bfloat16",
                    trust_remote_code=True,
                ),
                gen_config=dict(enable_thinking=self.config['enable_thinking']),
                peft_path=self.output_dir,
                max_out_len=self.config['max_out_len'],
                batch_size=self.batch_size,
                run_cfg=dict(num_gpus=1)
            )
        except Exception as e:
            logger.error("Model configuration failed: %s", str(e))
            raise

    def opencompass_configuration(self):
        """Setup opencompass configuration for evaluation."""
        logger.info("Setting up opencompass configuration...")
        
        # Create model directory
        try:
            model_name = self.config['model_name'].split('/')[1].split('-')[0].lower()
            config_dir = os.path.join(OPENCOMPASS_DIR, 'opencompass', 'configs', 'models', model_name)
            os.makedirs(config_dir, exist_ok=True)
        except Exception as e:
            logger.error("Model directory creation failed: %s", str(e))
            raise

        # Create model configuration file
        try:
            self.model_config = self._get_model_config()
            self.config_filename = f"{self.config['model_name'].split('/')[1]}.py"
            file_path = os.path.join(config_dir, self.config_filename)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('from opencompass.models import HuggingFacewithChatTemplate\n')
                file.write('import torch\n\n')
                file.write(f'models = [{self.model_config}]\n')
        except Exception as e:
            logger.error("Model configuration file creation failed: %s", str(e))
            raise
        
        logger.success("Opencompass configuration completed")

    def _get_latest_eval_dir(self):
        """Get the directory of the latest evaluation run."""
        try:
            eval_runs = [
                d for d in os.listdir(self.eval_dir)
                if os.path.isdir(os.path.join(self.eval_dir, d))
            ]
            if not eval_runs:
                raise FileNotFoundError("No evaluation runs found")
            
            # Get the latest evaluation run
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
            metric = row['metric']
            mode = row['mode']
            value = float(row[model_column]) if row[model_column] != '-' else None
            
            wandb.log({
                f"eval_{metric}/{dataset}": value
            })
            
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
            raise FileNotFoundError(f"No CSV files found in {self.summary_dir}")

        for filename in csv_files:
            filepath = os.path.join(self.summary_dir, filename)
            self._process_csv_file(filepath)
                
        logger.success("All results processed")

    def evaluation(self):
        """Run the evaluation process."""
        logger.info("Running evaluation...")
        opencompass_run = os.path.join(OPENCOMPASS_DIR, "run.py")
        
        try:
            cmd = [
                "python", opencompass_run,
                "--models", self.config_filename,
                *["--datasets"] + [dataset for dataset in self.config['eval_dataset']],
                "-w", self.eval_dir
            ]
            if self.resume:
                cmd.append("-r")
                
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            if "CUDA out of memory" in e.stderr:
                logger.warning("CUDA out of memory detected in subprocess.")
                logger.info(f"Current batch size: {self.batch_size}, try reducing to {self.batch_size - 1}")
                
                if self.batch_size <= 1:
                    logger.error("Batch size is 1, cannot reduce further, evaluation failed")
                    raise RuntimeError("Cannot find a suitable batch size to avoid OOM") from e
                
                self.batch_size -= 1
                self.resume = True
                logger.info("Reconfigure OpenCompass and retry evaluation...")
                self.opencompass_configuration()
                self.evaluation()
            else:
                logger.error("Evaluation failed: %s", e.stderr)
                raise
            
        logger.success("Evaluation completed")
    
    def run(self):
        """Run the complete evaluation pipeline."""
        self.opencompass_configuration()
        self.evaluation()
        self.summary_dir = self._get_latest_eval_dir()
        self._process_all_results()