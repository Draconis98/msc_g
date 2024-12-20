import logging
import wandb
from utils.config import ConfigManager
from training import TrainingPipeline
from evaluating import EvaluatingPipeline
from result_processor import ResultProcessor
from utils.misc import setup_logging
from utils.gpu import cleanup_gpu_memory

def pipeline():
    """Main pipeline execution."""
    
    # Initialize wandb
    try:
        wandb.init()
    except Exception as e:
        logging.error("Failed to initialize wandb: %s", str(e))
        raise
    
    # Setup logging
    setup_logging()
    
    # Create and validate configuration
    try:
        config = ConfigManager.create_config()
        ConfigManager.validate_config(config)
    except Exception as e:
        logging.error("Failed to create or validate configuration: %s", str(e))
        raise
    
    # Run training pipeline
    try:
        logging.info("Starting training pipeline...")
        training = TrainingPipeline(config)
        training.run()

        # NOTE: Don't know why but it's necessary to call this function again after training
        del training
        cleanup_gpu_memory()
    except Exception as e:
        logging.error("Failed to run training pipeline: %s", str(e))
        raise

    # Run evaluation pipeline
    try:
        logging.info("Starting evaluation pipeline...")
        evaluating = EvaluatingPipeline(config)
        evaluating.run()
    except Exception as e:
        logging.error("Failed to run evaluation pipeline: %s", str(e))
        raise
    
    # Process and log results
    try:
        logging.info("Processing evaluation results...")
        processor = ResultProcessor(config)
        processor.run()
    except Exception as e:
        logging.error("Failed to process evaluation results: %s", str(e))
        raise
    
    logging.info("Pipeline completed successfully")