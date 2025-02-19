from loguru import logger
import wandb
from utils.config import ConfigManager
from training import TrainingPipeline
from evaluating import EvaluatingPipeline
from result_processor import ResultProcessor
from utils.gpu import cleanup_gpu_memory

def pipeline():
    """Main pipeline execution."""
    
    # Initialize wandb
    try:
        wandb.init()
        logger.success("Wandb initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize wandb: %s", str(e))
        raise
    
    # Create and validate configuration
    try:
        config = ConfigManager.create_config()
        ConfigManager.validate_config(config)
    except Exception as e:
        logger.error("Failed to create or validate configuration: %s", str(e))
        raise
    
    # Run training pipeline
    try:
        logger.info("Starting training pipeline...")
        training = TrainingPipeline(config)
        training.run()

        # NOTE: Don't know why but it's necessary to call this function again after training
        del training
        cleanup_gpu_memory()
    except Exception as e:
        logger.error("Failed to run training pipeline: %s", str(e))
        raise

    # Run evaluation pipeline
    try:
        logger.info("Starting evaluation pipeline...")
        evaluating = EvaluatingPipeline(config)
        evaluating.run()
    except Exception as e:
        logger.error("Failed to run evaluation pipeline: %s", str(e))
        raise
    
    # Process and log results
    try:
        logger.info("Processing evaluation results...")
        processor = ResultProcessor(config)
        processor.run()
    except Exception as e:
        logger.error("Failed to process evaluation results: %s", str(e))
        raise
    
    logger.success("Pipeline completed successfully")