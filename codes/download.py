import os
import logging
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from utils.config import MODELS_DIR
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
            ignore_patterns=exclude_files,
            max_workers=1
        )
        logging.info(f"Successfully downloaded {model_id}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {model_id}: {str(e)}")
        return False

def download_models(model_ids, target_dir=None, exclude_files=None, resume=True, max_concurrent=3, parallel=True):
    """
    Download multiple models either concurrently or sequentially.
    
    Args:
        model_ids (list): List of model IDs to download
        target_dir (str, optional): Base target directory
        exclude_files (list, optional): List of file patterns to exclude
        resume (bool, optional): Whether to resume download if interrupted
        max_concurrent (int, optional): Maximum number of concurrent downloads
        parallel (bool, optional): Whether to download models in parallel
    """
    if parallel:
        logging.info(f"Downloading {len(model_ids)} models in parallel (max {max_concurrent} concurrent downloads)")
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            for model_id in model_ids:
                model_target_dir = os.path.join(target_dir, model_id) if target_dir else None
                future = executor.submit(download_model, model_id, model_target_dir, exclude_files, resume)
                futures.append((model_id, future))
            
            # Wait for all downloads to complete
            for model_id, future in futures:
                success = future.result()
                if success:
                    logging.info(f"Completed downloading {model_id}")
                else:
                    logging.error(f"Failed to download {model_id}")
    else:
        logging.info(f"Downloading {len(model_ids)} models sequentially")
        for model_id in model_ids:
            model_target_dir = os.path.join(target_dir, model_id) if target_dir else None
            success = download_model(model_id, model_target_dir, exclude_files, resume)
            if success:
                logging.info(f"Completed downloading {model_id}")
            else:
                logging.error(f"Failed to download {model_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download models from Hugging Face')
    parser.add_argument('-m', '--model-ids', type=str, required=True, 
                      help='Comma-separated list of model IDs (e.g., meta-llama/Llama-2-7b-chat-hf,facebook/opt-350m)')
    parser.add_argument('-t', '--target-dir', type=str, help='Base target directory for downloads', default=None)
    parser.add_argument('-e', '--except', dest='exclude_files', type=lambda x: x.split(','), 
                      help='Comma-separated list of file patterns to exclude', default=None)
    parser.add_argument('-r', '--resume', action='store_true', help='Resume download if interrupted', default=True)
    parser.add_argument('-c', '--concurrent', type=int, help='Maximum number of concurrent downloads', default=3)
    parser.add_argument('-p', '--parallel', action='store_true', help='Download models in parallel', default=False)
    
    args = parser.parse_args()
    
    # Split model IDs string into list
    model_ids = [mid.strip() for mid in args.model_ids.split(',')]
    
    download_models(model_ids, args.target_dir, args.exclude_files, args.resume, args.concurrent, args.parallel)
