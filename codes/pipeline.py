import logging
import wandb
from utils.config import ConfigManager
from training import TrainingPipeline
from evaluation_pipeline import EvaluationPipeline
from result_processor import ResultProcessor
from utils.misc import setup_logging

def main():
    """Main pipeline execution."""
    try:
        # Initialize wandb
        wandb.init()
        
        # Setup logging
        setup_logging()
        
        # Create and validate configuration
        config = ConfigManager.create_config()
        ConfigManager.validate_config(config)
        ConfigManager.log_config(config)
        
        # Run training pipeline
        logging.info("Starting training pipeline...")
        training = TrainingPipeline(config)
        output_dir = training.run()
        
        # Run evaluation pipeline
        logging.info("Starting evaluation pipeline...")
        evaluation = EvaluationPipeline(config, output_dir)
        summary_dir = evaluation.run()
        
        # Process and log results
        logging.info("Processing evaluation results...")
        processor = ResultProcessor(summary_dir)
        processor.process_all_results()
        processor.cleanup()
        
        logging.info("Pipeline completed successfully")
        
        try:
            wandb.finish()
        except Exception as e:
            logging.error(f"Failed to finish wandb run: {str(e)}")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()