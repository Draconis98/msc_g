import os
import logging
import subprocess
import wandb
import yaml
from utils.config import OPENCOMPASS_DIR

class EvaluatingPipeline:
    """Manages the complete evaluation process including setup, running, and results processing."""
    
    def __init__(self, config, output_dir):
        """Initialize evaluator with configuration and output directory."""
        self.config = config
        self.output_dir = output_dir
        self.eval_dir = os.path.join(output_dir, "eval")
        self.model_config = None
        self.config_filename = None
        
    def _get_model_config(self):
        """Generate model configuration for OpenCompass."""
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
            'max_out_len': 50,
            'max_seq_len': 1024,
            'batch_size': 32,
            'run_cfg': {'num_gpus': 1, 'num_procs': 1},
        }

    def setup(self):
        """Setup evaluation configuration and environment."""
        logging.info("Setting up evaluation...")
        
        # Create model directory
        model_name = self.config['model_name'].split(':')[0]
        model_name = ''.join([i for i in model_name if not i.isdigit()])
        eval_dir = os.path.join(OPENCOMPASS_DIR, 'opencompass', 'configs', 'models', model_name)
        os.makedirs(eval_dir, exist_ok=True)

        # Create model configuration file
        self.model_config = self._get_model_config()
        self.config_filename = f"{self.config['model_name'].replace(':', '-')}.py"
        file_path = os.path.join(eval_dir, self.config_filename)

        with open(file_path, 'w') as file:
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

        # Log evaluation configuration to wandb
        wandb.log({
            "evaluation_config": {
                "model_name": self.config['model_name'],
                "strategy": self.config['strategy'],
                "config_path": file_path,
                "model_config": self.model_config
            }
        })
        
        logging.info("Evaluation setup completed")

    def run_evaluation(self):
        """Run the evaluation process."""
        logging.info("Running evaluation...")
        opencompass_run = os.path.join(OPENCOMPASS_DIR, "run.py")
        
        try:
            result = subprocess.run([
                "python", opencompass_run,
                "--models", self.config_filename,
                "--datasets", self.config['eval_dataset'] + "_gen",
                "-w", self.eval_dir
            ], check=True, capture_output=True, text=True, encoding="utf-8")
            logging.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Evaluation failed: {e.stderr}")
            raise
            
        logging.info("Evaluation completed")
        
    def get_latest_run_dir(self):
        """Get the directory of the latest evaluation run."""
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

    def process_evaluation_results(self, results_path):
        """Process and log evaluation results to wandb."""
        if not os.path.exists(results_path):
            wandb.log({"evaluation_status": "No results file found"})
            return None

        try:
            with open(results_path, 'r') as f:
                results = yaml.safe_load(f)
                
            # Process and structure results
            processed_results = {
                "overall_score": results.get("overall_score", 0),
                "task_scores": results.get("task_scores", {}),
                "completion_time": results.get("completion_time", 0)
            }
            
            # Log to wandb
            wandb.log({
                "evaluation_results": processed_results,
                "evaluation_status": "completed"
            })
            
            return processed_results
            
        except Exception as e:
            wandb.log({
                "evaluation_error": str(e),
                "evaluation_status": "failed"
            })
            return None
    
    def run(self):
        """Run the complete evaluation pipeline."""
        try:
            self.setup()
            self.run_evaluation()
            return self.get_latest_run_dir()
        except Exception as e:
            logging.error(f"Evaluation pipeline failed: {str(e)}")
            raise