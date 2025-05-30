"""Pipeline module for running the training and evaluation pipelines.

This module is responsible for running the training and evaluation pipelines.
It initializes wandb, creates and validates the configuration,
runs the training pipeline, runs the evaluation pipeline,
and processes the evaluation results.
"""

import gc

import torch
import wandb
from loguru import logger

from utils.config import ConfigManager
from training import TrainingPipeline
from evaluating import LLMEvaluatingPipeline

def cleanup_gpu_memory():
    """Clean up GPU memory by forcing garbage collection and emptying cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def pipeline():
    """Main pipeline execution."""
    
    # Initialize wandb
    try:
        wandb.init()
        logger.success("Wandb initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        raise
    
    # Create and validate configuration
    try:
        config = ConfigManager.create_config()
        ConfigManager.validate_config(config)
    except Exception as e:
        logger.error(f"Failed to create or validate configuration: {e}")
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
        logger.error(f"Failed to run training pipeline: {e}")
        raise

    # Run evaluation pipeline
    try:
        logger.info("Starting evaluation pipeline...")
        evaluating = LLMEvaluatingPipeline(config)
        evaluating.run()
    except Exception as e:
        logger.error(f"Failed to run evaluation pipeline: {e}")
        raise
    
    logger.success("Pipeline completed successfully")