import os
import csv
import logging
import wandb

class ResultProcessor:
    """Processes and logs evaluation results."""
    
    def __init__(self, summary_dir):
        """Initialize result processor."""
        self.summary_dir = summary_dir
        
    def process_csv_file(self, filepath):
        """Process a single CSV file and log results to wandb."""
        try:
            with open(filepath, 'r') as f:
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
                        logging.warning(f"Error processing row in {filepath}: {str(e)}")
                        
                logging.info(f"Processed {processed_rows} rows from {filepath}")
                
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {str(e)}")
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
            raise KeyError(f"Missing required column: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Invalid value in {model_column} column: {str(e)}")
            
    def process_all_results(self):
        """Process all CSV files in the summary directory."""
        logging.info(f"Processing results from {self.summary_dir}")
        
        try:
            csv_files = [
                f for f in os.listdir(self.summary_dir)
                if f.endswith('.csv')
            ]
            
            if not csv_files:
                raise RuntimeError(f"No CSV files found in {self.summary_dir}")
                
            for filename in csv_files:
                filepath = os.path.join(self.summary_dir, filename)
                self.process_csv_file(filepath)
                
            logging.info("All results processed successfully")
            
        except Exception as e:
            logging.error(f"Error processing results: {str(e)}")
            raise
            
    def cleanup(self):
        """Cleanup and finalize result processing."""
        try:
            wandb.finish()
            logging.info("Results processing completed and wandb session finished")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            raise 