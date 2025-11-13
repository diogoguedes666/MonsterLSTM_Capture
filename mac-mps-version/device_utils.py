"""
Device utilities for cross-platform GPU support (MPS/CUDA/CPU).

This module provides a unified interface for device management across
different backends: Apple Metal Performance Shaders (MPS), NVIDIA CUDA, and CPU.
"""

import torch
from typing import Optional, Dict, Any, Tuple


def get_device(force_cpu: bool = False, cuda_preferred: bool = False) -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Priority order (unless overridden):
    1. MPS (Apple Silicon GPU) - if available and not forcing CPU
    2. CUDA (NVIDIA GPU) - if available and not forcing CPU
    3. CPU - fallback
    
    Args:
        force_cpu: If True, always return CPU device
        cuda_preferred: If True, prefer CUDA over MPS when both are available
        
    Returns:
        torch.device: The best available device
    """
    if force_cpu:
        return torch.device('cpu')
    
    # Check CUDA first if preferred
    if cuda_preferred and torch.cuda.is_available():
        return torch.device('cuda')
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Check CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Fallback to CPU
    return torch.device('cpu')


def get_device_name(device: Optional[torch.device] = None) -> str:
    """
    Get a human-readable name for the device.
    
    Args:
        device: The device to get name for. If None, uses get_device()
        
    Returns:
        str: Device name
    """
    if device is None:
        device = get_device()
    
    if device.type == 'mps':
        return 'Apple Silicon GPU (MPS)'
    elif device.type == 'cuda':
        if torch.cuda.is_available():
            return f'NVIDIA {torch.cuda.get_device_name(0)}'
        return 'CUDA GPU'
    else:
        return 'CPU'


def get_memory_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get memory information for the device.
    
    Args:
        device: The device to get memory info for. If None, uses get_device()
        
    Returns:
        dict: Memory information with keys:
            - total: Total memory in GB (if available)
            - allocated: Allocated memory in GB (if available)
            - cached: Cached memory in GB (if available)
            - free: Free memory in GB (if available)
            - available: Whether memory info is available
    """
    if device is None:
        device = get_device()
    
    info = {
        'total': None,
        'allocated': None,
        'cached': None,
        'free': None,
        'available': False
    }
    
    if device.type == 'cuda' and torch.cuda.is_available():
        info['total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
        info['cached'] = torch.cuda.memory_reserved(0) / (1024**3)
        info['free'] = info['total'] - info['cached']
        info['available'] = True
    elif device.type == 'mps':
        # MPS doesn't provide detailed memory info, but we can check if it's available
        info['available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        # MPS uses unified memory, so we can't get exact GPU memory stats
        info['total'] = None
        info['allocated'] = None
        info['cached'] = None
        info['free'] = None
    else:
        # CPU - no GPU memory info
        info['available'] = False
    
    return info


def clear_cache(device: Optional[torch.device] = None) -> None:
    """
    Clear the cache for the device.
    
    Args:
        device: The device to clear cache for. If None, uses get_device()
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS doesn't have explicit cache clearing, but we can try
        if hasattr(torch.backends, 'mps'):
            torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    # CPU doesn't need cache clearing


def supports_amp(device: Optional[torch.device] = None) -> bool:
    """
    Check if the device supports automatic mixed precision (AMP).
    
    Args:
        device: The device to check. If None, uses get_device()
        
    Returns:
        bool: True if AMP is supported (currently only CUDA)
    """
    if device is None:
        device = get_device()
    
    # Currently only CUDA supports torch.cuda.amp.autocast()
    # MPS may support it in future PyTorch versions
    return device.type == 'cuda' and torch.cuda.is_available()


def get_amp_context(device: Optional[torch.device] = None):
    """
    Get the appropriate AMP context manager for the device.
    
    Args:
        device: The device to get AMP context for. If None, uses get_device()
        
    Returns:
        Context manager: AMP context if supported, otherwise nullcontext
    """
    from contextlib import nullcontext
    
    if device is None:
        device = get_device()
    
    if supports_amp(device):
        return torch.cuda.amp.autocast()
    else:
        return nullcontext()


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """
    Reset peak memory statistics for the device.
    
    Args:
        device: The device to reset stats for. If None, uses get_device()
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_allocated(device: Optional[torch.device] = None) -> Optional[float]:
    """
    Get peak memory allocated in GB.
    
    Args:
        device: The device to get peak memory for. If None, uses get_device()
        
    Returns:
        float: Peak memory in GB, or None if not available
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda' and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(0) / (1024**3)
    # MPS doesn't provide peak memory stats
    return None


def move_to_device(tensor_or_model, device: Optional[torch.device] = None):
    """
    Move tensor or model to the specified device.
    
    Args:
        tensor_or_model: Tensor or model to move
        device: Target device. If None, uses get_device()
        
    Returns:
        Tensor or model on the target device
    """
    if device is None:
        device = get_device()
    
    return tensor_or_model.to(device)


def print_device_info(device: Optional[torch.device] = None) -> None:
    """
    Print detailed information about the device.
    
    Args:
        device: The device to print info for. If None, uses get_device()
    """
    if device is None:
        device = get_device()
    
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"Device Type: {device.type.upper()}")
    print(f"Device Name: {get_device_name(device)}")
    
    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    mem_info = get_memory_info(device)
    if mem_info['available']:
        print(f"\nMemory Information:")
        if mem_info['total'] is not None:
            print(f"  Total: {mem_info['total']:.2f} GB")
        if mem_info['allocated'] is not None:
            print(f"  Allocated: {mem_info['allocated']:.2f} GB")
        if mem_info['cached'] is not None:
            print(f"  Cached: {mem_info['cached']:.2f} GB")
        if mem_info['free'] is not None:
            print(f"  Free: {mem_info['free']:.2f} GB")
    else:
        if device.type == 'mps':
            print("\nMemory: Unified Memory (shared with system)")
        else:
            print("\nMemory: CPU RAM")
    
    print(f"\nAMP Support: {'Yes' if supports_amp(device) else 'No'}")
    print("=" * 60)






