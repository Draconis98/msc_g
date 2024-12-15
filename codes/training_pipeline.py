import os
import logging
import wandb
from utils.misc import create_run_name, ensure_dir
from data_processor import load_and_process_data
from trainer import setup_training, train_model, get_output_dir

class TrainingPipeline:
    """Manages the training pipeline."""
    
    def __init__(self, config):
        """Initialize training pipeline with configuration."""
        self.config = config
        self.output_dir = None
        self.tokenizer = None
        self.processed_dataset = None
        self.trainer = None
        
    def setup(self):
        """Setup training environment and resources."""
        # Setup wandb run name
        wandb.run.name = create_run_name(self.config)
        
        # Setup output directory
        self.output_dir = get_output_dir(self.config)
        ensure_dir(os.path.join(self.output_dir, "eval"))
        
        logging.info("Training pipeline setup completed")
        
    def prepare_data(self):
        """Load and process training data."""
        logging.info(f"Loading and processing data for {self.config['dataset']}...")
        self.tokenizer, self.processed_dataset = load_and_process_data(self.config)
        logging.info("Data preparation completed")
        
    def setup_trainer(self):
        """Setup the trainer."""
        logging.info("Setting up trainer...")
        self.trainer = setup_training(
            self.config,
            self.tokenizer,
            self.processed_dataset,
            self.output_dir
        )
        logging.info("Trainer setup completed")
        
    def train(self):
        """Run the training process."""
        logging.info("Starting training...")
        train_model(self.trainer, self.output_dir, self.tokenizer)
        logging.info("Training completed")
        
    def run(self):
        """Run the complete training pipeline."""
        try:
            self.setup()
            self.prepare_data()
            self.setup_trainer()
            self.train()
            return self.output_dir
        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise 