import torch
import gc

def is_gpu_free(gpu_id):
    """Check if a specific GPU is free."""
    if not torch.cuda.is_available():
        return False
    gpu_memory = torch.cuda.memory_reserved(gpu_id)
    return gpu_memory == 0

def get_available_gpus():
    """Get a list of available GPU IDs."""
    available_gpus = []
    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            if is_gpu_free(gpu_id):
                available_gpus.append(gpu_id)
    return available_gpus

def cleanup_gpu_memory():
    """Clean up GPU memory by forcing garbage collection and emptying cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

def set_gpu_device(gpu_id):
    """Set the GPU device for computation."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return True
    return False

def get_gpu_memory_info(gpu_id=None):
    """Get memory information for specified GPU or all GPUs."""
    if not torch.cuda.is_available():
        return None
    
    if gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            return None
        return {
            'total': torch.cuda.get_device_properties(gpu_id).total_memory,
            'reserved': torch.cuda.memory_reserved(gpu_id),
            'allocated': torch.cuda.memory_allocated(gpu_id)
        }
    
    return {i: get_gpu_memory_info(i) for i in range(torch.cuda.device_count())} 