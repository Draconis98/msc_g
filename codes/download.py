import os
import logging
import argparse
import sys

from config import MODELS_DIR
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Check if huggingface_hub is installed
try:
    from huggingface_hub import snapshot_download
except ImportError:
    logging.warning("Package 'huggingface_hub' is not installed.")
    while True:
        response = input("Would you like to install it now? (y/n): ").lower().strip()
        if response in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'")
    
    if response == 'y':
        import subprocess
        try:
            logging.info("Installing huggingface_hub...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            from huggingface_hub import snapshot_download
            logging.info("Successfully installed huggingface_hub")
        except subprocess.CalledProcessError as e:
            logging.error("Failed to install huggingface_hub. Please install it manually using:")
            logging.error("pip install huggingface_hub")
            sys.exit(1)
    else:
        logging.info("Please install huggingface_hub manually using:")
        logging.info("pip install huggingface_hub")
        sys.exit(1)

def download_model(model_id, target_dir=None, exclude_files=None, resume=True):
    """
    Download model from Hugging Face mirror to specified location.
    
    Args:
        model_id (str): Full model ID from Hugging Face (e.g., 'meta-llama/Llama-2-7b-chat-hf')
        target_dir (str, optional): Target directory to download to. If None, will use MODELS_DIR/model_name
        exclude_files (list, optional): List of file patterns to exclude from download
        resume (bool, optional): Whether to resume download if interrupted
    """
    
    if target_dir is None:
        target_dir = os.path.join(MODELS_DIR, model_id)
    
    os.makedirs(target_dir, exist_ok=True)
    
    logging.info(f"Downloading {model_id} to {target_dir}")
    
    try:
        # Download using Hugging Face Hub with mirror
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            endpoint="https://hf-mirror.com",
            ignore_patterns=exclude_files
        )
        logging.info(f"Successfully downloaded {model_id}")
        
    except Exception as e:
        logging.error(f"Error downloading {model_id}: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download model from Hugging Face')
    parser.add_argument('-m', '--model-id', type=str, required=True, help='Full model ID from Hugging Face (e.g., meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('-t', '--target-dir', type=str, help='Target directory to download to', default=None)
    parser.add_argument('-e', '--except', dest='exclude_files', type=lambda x: x.split(','), help='Comma-separated list of file patterns to exclude', default=None)
    parser.add_argument('-r', '--resume', action='store_true', help='Resume download if interrupted', default=True)
    
    args = parser.parse_args()
    
    download_model(args.model_id, args.target_dir, args.exclude_files, args.resume)
