import os
import wandb
from config import OPENCOMPASS_DIR

def get_model_config(config, output_dir):
    """Generate model configuration for OpenCompass."""
    return {
        'abbr': f"{config['model_name'].replace(':', '-')}-{config['strategy']}",
        'type': 'HuggingFacewithChatTemplate',
        'path': output_dir,
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
        'batch_size': 16,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    }

def setup_evaluator(config, output_dir):
    """Setup evaluation configuration for OpenCompass."""
    # Create model directory
    model_name = config['model_name'].split(':')[0]
    model_name = ''.join([i for i in model_name if not i.isdigit()])
    eval_dir = os.path.join(OPENCOMPASS_DIR, 'opencompass', 'configs', 'models', model_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Create model configuration file
    model_config = get_model_config(config, output_dir)
    config_filename = f"{config['model_name'].replace(':', '-')}.py"
    file_path = os.path.join(eval_dir, config_filename)

    with open(file_path, 'w') as file:
        file.write('from opencompass.models import HuggingFacewithChatTemplate\n')
        file.write('import torch\n\n')
        file.write('models = [\n')
        file.write('    dict(\n')
        for key, value in model_config.items():
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
            "model_name": config['model_name'],
            "strategy": config['strategy'],
            "config_path": file_path,
            "model_config": model_config
        }
    })

    return config_filename

def process_evaluation_results(results_path):
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