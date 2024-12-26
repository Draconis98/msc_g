"""Module for processing and logging evaluation results to Weights & Biases."""

import os
import csv
import logging
import wandb
from utils.misc import get_output_dir

class ResultProcessor:
    """Processes and logs evaluation results."""
    
    def __init__(self, config):
        """Initialize result processor."""
        self.config = config
        self.output_dir = get_output_dir(config)
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.summary_dir = None

    def _get_latest_eval_dir(self):
        """Get the directory of the latest evaluation run."""
        try:
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
        except Exception as e:
            logging.error("Error getting latest run directory: %s", str(e))
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
                        logging.warning("Error processing row in %s: %s" % (filepath, str(e)))
                        
                logging.info("Processed %d rows from %s" % (processed_rows, filepath))
                
        except Exception as e:
            logging.error("Error processing file %s: %s" % (filepath, str(e)))
            raise
            
    def _log_row_to_wandb(self, row, model_column):
        """Log a single row of results to wandb."""
        try:
            dataset = row['dataset']
            version = row['version']
            metric = row['metric']
            mode = row['mode']
            value = float(row[model_column])
            
            metric_name = f"eval/{dataset}/{version}/{metric}/{mode}"
            wandb.log({metric_name: value})
            
        except KeyError as e:
            raise KeyError(f"Missing required column: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Invalid value in {model_column} column: {str(e)}") from e
            
    def _process_all_results(self):
        """Process all CSV files in the summary directory."""
        logging.info("Processing results from %s", self.summary_dir)
        
        try:
            csv_files = [
                f for f in os.listdir(self.summary_dir)
                if f.endswith('.csv')
            ]
        except Exception as e:
            logging.error("Error processing results: %s" % str(e))
            raise
            
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {self.summary_dir}")

        for filename in csv_files:
            filepath = os.path.join(self.summary_dir, filename)
            self._process_csv_file(filepath)
                
        logging.info("All results processed successfully")

    def run(self):
        """Run the result processing pipeline."""
        self.summary_dir = self._get_latest_eval_dir()
        self._process_all_results()
