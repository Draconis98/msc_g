import os
import logging
from evaluator import setup_evaluator
from config import OPENCOMPASS_DIR

class EvaluationPipeline:
    """Manages the evaluation pipeline."""
    
    def __init__(self, config, output_dir):
        """Initialize evaluation pipeline."""
        self.config = config
        self.output_dir = output_dir
        self.evaluator = None
        self.eval_dir = os.path.join(output_dir, "eval")
        
    def setup(self):
        """Setup evaluation environment."""
        logging.info("Setting up evaluation...")
        self.evaluator = setup_evaluator(self.config, self.output_dir)
        logging.info("Evaluation setup completed")
        
    def run_evaluation(self):
        """Run the evaluation process."""
        logging.info("Running evaluation...")
        opencompass_run = os.path.join(OPENCOMPASS_DIR, "run.py")
        eval_cmd = (
            f"python {opencompass_run} "
            f"--models {self.evaluator} "
            f"--datasets {self.config['eval_dataset']}_gen "
            f"-w {self.eval_dir}"
        )
        
        # Run evaluation command
        exit_code = os.system(eval_cmd)
        if exit_code != 0:
            raise RuntimeError(f"Evaluation failed with exit code {exit_code}")
            
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
        
    def run(self):
        """Run the complete evaluation pipeline."""
        try:
            self.setup()
            self.run_evaluation()
            return self.get_latest_run_dir()
        except Exception as e:
            logging.error(f"Evaluation pipeline failed: {str(e)}")
            raise 