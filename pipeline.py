"""Pipeline module for running the training and evaluation pipelines.

This module is responsible for running the training and evaluation pipelines.
It initializes wandb, creates and validates the configuration,
runs the training pipeline, runs the evaluation pipeline,
and processes the evaluation results.
"""

import gc
import random
import numpy as np
import os

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

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set seed to {seed}")

def pipeline(resume=False, run_id=None, debug=False):
    """Main pipeline execution."""
    
    # Initialize wandb
    if not resume:
        try:
            wandb.init()
            logger.success("Wandb initialized")
        except Exception as e:
            logger.error(f"Wandb initialization failed: {e}")
            raise
        
        # Create and validate configuration
        try:
            config = ConfigManager.create_config()
        except Exception as e:
            logger.error(f"Configuration creation failed: {e}")
            raise
    else:
        try:
            entity, project, id = map(str, run_id.split("/"))
            wandb.init(entity=entity, project=project, id=id, resume="must")
            logger.success("Wandb resumed")
            config = wandb.Api().run(run_id).config
        except Exception as e:
            logger.error(f"Wandb resume failed: {e}")
            raise
    
    # Setup seed
    setup_seed(config['seed'])
    
    # Run training pipeline
    try:
        logger.info("Starting training pipeline...")
        training = TrainingPipeline(config, resume=resume, debug=debug)
        training.run()

        # NOTE: Don't know why but it's necessary to call this function again after training
        del training
        cleanup_gpu_memory()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

    # Run evaluation pipeline
    try:
        logger.info("Starting evaluation pipeline...")
        evaluating = LLMEvaluatingPipeline(config, resume=resume)
        evaluating.run()
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        raise
    
    logger.success("Pipeline completed")