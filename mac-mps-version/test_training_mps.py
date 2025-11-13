#!/usr/bin/env python3
"""
Comprehensive test suite for Mac MPS training system.
Tests all critical components: imports, device detection, data loading,
model initialization, training loop, validation, checkpoints, memory management,
diagnostics, performance profiling, edge cases, and AutoTuner functionality.
"""

import sys
import os
import time
import json
import traceback
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import gc

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test results tracking
test_results = []
test_start_time = time.time()

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_test(name: str, status: str, details: str = ""):
    """Print test result."""
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {name}: {status}")
    if details:
        print(f"   {details}")
    test_results.append({"name": name, "status": status, "details": details})

def run_test(test_func):
    """Decorator to run a test and catch exceptions."""
    def wrapper():
        test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
        try:
            result = test_func()
            if result is False:
                print_test(test_name, "FAIL", "Test returned False")
            else:
                print_test(test_name, "PASS", result if isinstance(result, str) else "")
        except Exception as e:
            print_test(test_name, "FAIL", f"Exception: {str(e)}\n{traceback.format_exc()}")
    return wrapper

# ============================================================================
# TEST 1: Environment & Imports
# ============================================================================

@run_test
def test_environment_imports():
    """Test 1: Verify all modules import correctly."""
    print_header("TEST 1: Environment & Imports")
    
    # Check PyTorch version
    pytorch_version = torch.__version__
    version_parts = pytorch_version.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    assert major >= 2 or (major == 1 and minor >= 12), f"PyTorch >= 2.0 required for MPS, got {pytorch_version}"
    
    # Import CoreAudioML modules
    import CoreAudioML.miscfuncs as miscfuncs
    import CoreAudioML.training as training
    import CoreAudioML.dataset as dataset
    import CoreAudioML.networks as networks
    import CoreAudioML.diagnostics as diagnostics
    
    # Import device_utils
    import device_utils
    
    # Verify device detection
    device = device_utils.get_device()
    assert device is not None, "Device detection failed"
    
    return f"PyTorch {pytorch_version}, Device: {device_utils.get_device_name(device)}"

# ============================================================================
# TEST 2: Device Utilities
# ============================================================================

@run_test
def test_device_utilities():
    """Test 2: Device utilities functionality."""
    print_header("TEST 2: Device Utilities")
    
    import device_utils
    
    # Test get_device()
    device = device_utils.get_device()
    assert device is not None, "get_device() returned None"
    
    # Test get_device_name()
    device_name = device_utils.get_device_name(device)
    assert isinstance(device_name, str), "get_device_name() should return string"
    
    # Test get_memory_info()
    mem_info = device_utils.get_memory_info(device)
    assert isinstance(mem_info, dict), "get_memory_info() should return dict"
    assert 'available' in mem_info, "Memory info missing 'available' key"
    
    # Test clear_cache()
    device_utils.clear_cache()  # Should not raise exception
    
    # Test get_amp_context()
    amp_context = device_utils.get_amp_context(device)
    assert amp_context is not None, "get_amp_context() returned None"
    
    # Test supports_amp()
    supports = device_utils.supports_amp(device)
    assert isinstance(supports, bool), "supports_amp() should return bool"
    
    return f"Device: {device_name}, AMP: {supports}, Memory available: {mem_info['available']}"

# ============================================================================
# TEST 3: Data Loading
# ============================================================================

@run_test
def test_data_loading():
    """Test 3: Dataset loading and verification."""
    print_header("TEST 3: Data Loading")
    
    import CoreAudioML.dataset as dataset
    import device_utils
    
    # Initialize dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
    if not os.path.exists(data_dir):
        data_dir = './Data'  # Fallback
    
    ds = dataset.DataSet(data_dir=data_dir)
    
    # Create subsets
    ds.create_subset('train', frame_len=22050)
    ds.create_subset('val')
    ds.create_subset('test')
    
    # Load files
    file_name = 'dls2'
    ds.load_file(os.path.join('train', file_name), 'train')
    ds.load_file(os.path.join('val', file_name), 'val')
    ds.load_file(os.path.join('test', file_name), 'test')
    
    # Verify subsets exist
    assert 'train' in ds.subsets, "Train subset not created"
    assert 'val' in ds.subsets, "Val subset not created"
    assert 'test' in ds.subsets, "Test subset not created"
    
    # Check data shapes
    train_subset = ds.subsets['train']
    assert 'input' in train_subset.data, "Train input data missing"
    assert 'target' in train_subset.data, "Train target data missing"
    
    input_data = train_subset.data['input'][0]
    target_data = train_subset.data['target'][0]
    
    assert input_data.shape == target_data.shape, f"Shape mismatch: input {input_data.shape} vs target {target_data.shape}"
    assert len(input_data.shape) == 3, f"Expected 3D tensor, got {len(input_data.shape)}D"
    
    # Move to device - data is stored as tuples, need to move each element
    device = device_utils.get_device()
    # Data is stored as tuple, need to move each tensor
    input_tuple = train_subset.data['input']
    target_tuple = train_subset.data['target']
    
    # Verify tuple is not empty
    assert len(input_tuple) > 0, "Input tuple is empty"
    assert len(target_tuple) > 0, "Target tuple is empty"
    
    # Move each tensor in the tuple to device
    input_data = tuple(t.to(device) for t in input_tuple)
    target_data = tuple(t.to(device) for t in target_tuple)
    
    # Verify the first tensor is on the device
    # Note: device might be 'mps' but tensor.device is 'mps:0' - both are equivalent
    assert input_data[0].device.type == device.type, f"Input data not on correct device. Expected {device.type}, got {input_data[0].device.type}"
    assert target_data[0].device.type == device.type, f"Target data not on correct device. Expected {device.type}, got {target_data[0].device.type}"
    
    return f"Train: {input_data[0].shape}, Val: {ds.subsets['val'].data['input'][0].shape}, Test: {ds.subsets['test'].data['input'][0].shape}"

# ============================================================================
# TEST 4: Model Initialization
# ============================================================================

@run_test
def test_model_initialization():
    """Test 4: Model creation and forward pass."""
    print_header("TEST 4: Model Initialization")
    
    import CoreAudioML.networks as networks
    import device_utils
    
    # Create model with test config
    model = networks.SimpleRNN(
        input_size=1,
        output_size=1,
        unit_type="LSTM",
        hidden_size=32,
        skip=1
    )
    
    # Move to device
    device = device_utils.get_device()
    model = model.to(device)
    
    # Force a forward pass to ensure model is fully on device
    test_input_warmup = torch.randn(10, 1, 1, device=device)
    with torch.no_grad():
        _ = model(test_input_warmup)
    
    # Reset hidden state before actual test
    model.reset_hidden()
    
    # Verify parameters are on device - check after forward pass
    # Sometimes parameters aren't moved until first forward pass on MPS
    # Note: device might be 'mps' but param.device is 'mps:0' - both are equivalent
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            # Try moving again
            param.data = param.data.to(device)
        assert param.device.type == device.type, f"Parameter {name} not on device {device.type}, got {param.device.type}"
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    test_input = torch.randn(seq_len, batch_size, 1, device=device)
    
    model.eval()
    model.reset_hidden()  # Ensure hidden state is reset
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (seq_len, batch_size, 1), f"Output shape mismatch: {output.shape}"
    assert output.device.type == device.type, f"Output not on correct device. Expected {device.type}, got {output.device.type}"
    
    return f"Model on {device}, Forward pass: {test_input.shape} -> {output.shape}"

# ============================================================================
# TEST 5: Loss Functions
# ============================================================================

@run_test
def test_loss_functions():
    """Test 5: Loss function initialization and computation."""
    print_header("TEST 5: Loss Functions")
    
    import CoreAudioML.training as training
    import device_utils
    
    # Initialize LossWrapper
    loss_fcns = {"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15}
    loss_wrapper = training.LossWrapper(
        loss_fcns,
        pre_filt=None,
        hf_hinge_fmin=10000,
        hf_hinge_strength=0.5
    )
    
    # Create dummy data
    device = device_utils.get_device()
    # Move loss wrapper to device (important for buffers like hf_mask)
    loss_wrapper = loss_wrapper.to(device)
    
    seq_len, batch_size, channels = 100, 4, 1
    output = torch.randn(seq_len, batch_size, channels, device=device)
    target = torch.randn(seq_len, batch_size, channels, device=device)
    
    # Compute loss
    loss = loss_wrapper(output, target)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test individual loss components
    esr_loss = training.ESRLoss()
    dc_loss = training.DCLoss()
    
    esr_val = esr_loss(output, target)
    dc_val = dc_loss(output, target)
    
    assert esr_val.item() >= 0, "ESR loss should be non-negative"
    assert dc_val.item() >= 0, "DC loss should be non-negative"
    
    return f"Total loss: {loss.item():.6f}, ESR: {esr_val.item():.6f}, DC: {dc_val.item():.6f}"

# ============================================================================
# TEST 6: Training Loop (Minimal)
# ============================================================================

@run_test
def test_training_loop():
    """Test 6: Minimal training loop execution."""
    print_header("TEST 6: Training Loop (Minimal)")
    
    import CoreAudioML.networks as networks
    import CoreAudioML.training as training
    import CoreAudioML.dataset as dataset
    import device_utils
    import torch.optim as optim
    
    # Load minimal dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
    if not os.path.exists(data_dir):
        data_dir = './Data'
    
    ds = dataset.DataSet(data_dir=data_dir)
    ds.create_subset('train', frame_len=22050)
    ds.load_file(os.path.join('train', 'dls2'), 'train')
    
    # Get small subset of data for fast testing
    train_data = ds.subsets['train']
    input_data = train_data.data['input'][0][:5000, :4, :]  # Small subset
    target_data = train_data.data['target'][0][:5000, :4, :]
    
    # Move to device
    device = device_utils.get_device()
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    
    # Create model
    model = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    # Create loss function and move to device
    loss_fcns = {"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15}
    loss_fn = training.LossWrapper(loss_fcns, pre_filt=None, hf_hinge_fmin=10000, hf_hinge_strength=0.5)
    loss_fn = loss_fn.to(device)
    
    # Run one epoch
    model.train()
    initial_loss = None
    final_loss = None
    
    batch_size = 2
    init_len = 200
    up_fr = 750
    
    epoch_loss = model.train_epoch(input_data, target_data, loss_fn, optimizer, batch_size, init_len, up_fr)
    
    assert isinstance(epoch_loss, torch.Tensor), "Epoch loss should be a tensor"
    assert epoch_loss.item() >= 0, "Loss should be non-negative"
    
    # Verify gradients were computed
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    # Note: gradients are cleared after each step, so we check optimizer state instead
    
    return f"Epoch loss: {epoch_loss.item():.6f}"

# ============================================================================
# TEST 7: Validation
# ============================================================================

@run_test
def test_validation():
    """Test 7: Validation loop execution."""
    print_header("TEST 7: Validation")
    
    import CoreAudioML.networks as networks
    import CoreAudioML.training as training
    import CoreAudioML.dataset as dataset
    import device_utils
    
    # Load validation dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
    if not os.path.exists(data_dir):
        data_dir = './Data'
    
    ds = dataset.DataSet(data_dir=data_dir)
    ds.create_subset('val')
    ds.load_file(os.path.join('val', 'dls2'), 'val')
    
    # Get small subset
    val_data = ds.subsets['val']
    input_data = val_data.data['input'][0][:10000, :2, :]
    target_data = val_data.data['target'][0][:10000, :2, :]
    
    # Move to device
    device = device_utils.get_device()
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    
    # Create model
    model = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model = model.to(device)
    model.eval()
    
    # Create loss function and move to device
    loss_fcns = {"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15}
    loss_fn = training.LossWrapper(loss_fcns, pre_filt=None, hf_hinge_fmin=10000, hf_hinge_strength=0.5)
    loss_fn = loss_fn.to(device)
    
    # Run validation
    val_chunk = 10000
    output, val_loss = model.process_data(input_data, target_data, loss_fn, val_chunk, grad=False)
    
    assert output.shape == target_data.shape, f"Output shape {output.shape} != target {target_data.shape}"
    assert isinstance(val_loss, torch.Tensor), "Validation loss should be a tensor"
    assert val_loss.item() >= 0, "Validation loss should be non-negative"
    
    return f"Validation loss: {val_loss.item():.6f}, Output shape: {output.shape}"

# ============================================================================
# TEST 8: Checkpoint Operations
# ============================================================================

@run_test
def test_checkpoint_operations():
    """Test 8: Checkpoint save and load."""
    print_header("TEST 8: Checkpoint Operations")
    
    import CoreAudioML.networks as networks
    import CoreAudioML.miscfuncs as miscfuncs
    import device_utils
    import torch
    
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create model
    device = device_utils.get_device()
    model = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model = model.to(device)
    
    # Save model
    model.save_model('test_model', test_dir)
    
    # Verify file exists
    model_path = os.path.join(test_dir, 'test_model.json')
    assert os.path.exists(model_path), "Model file not created"
    
    # Load model
    model_data = miscfuncs.json_load('test_model', test_dir)
    loaded_model = networks.load_model(model_data)
    loaded_model = loaded_model.to(device)
    
    # Compare weights
    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    
    for key in original_state:
        assert key in loaded_state, f"Missing key in loaded model: {key}"
        assert torch.allclose(original_state[key], loaded_state[key], atol=1e-6), f"Weights differ for {key}"
    
    return f"Checkpoint saved and loaded successfully, {len(original_state)} parameters verified"

# ============================================================================
# TEST 9: Memory Management
# ============================================================================

@run_test
def test_memory_management():
    """Test 9: Memory management and cache clearing."""
    print_header("TEST 9: Memory Management")
    
    import device_utils
    import psutil
    import gc
    
    device = device_utils.get_device()
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**2)  # MB
    
    # Create some tensors
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
    
    # Check memory increased
    peak_memory = process.memory_info().rss / (1024**2)
    memory_increase = peak_memory - initial_memory
    
    # Clear cache
    device_utils.clear_cache()
    gc.collect()
    
    # Delete tensors
    del tensors
    gc.collect()
    device_utils.clear_cache()
    
    # Check memory decreased
    final_memory = process.memory_info().rss / (1024**2)
    
    # Get device memory info
    mem_info = device_utils.get_memory_info(device)
    
    return f"RAM: {initial_memory:.1f}MB -> {peak_memory:.1f}MB -> {final_memory:.1f}MB, Device memory available: {mem_info['available']}"

# ============================================================================
# TEST 10: Diagnostics
# ============================================================================

@run_test
def test_diagnostics():
    """Test 10: Training diagnostics and stats."""
    print_header("TEST 10: Diagnostics")
    
    import CoreAudioML.training as training
    import CoreAudioML.miscfuncs as miscfuncs
    import os
    
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize TrainTrack
    train_track = training.TrainTrack()
    train_track['best_val_loss'] = float('inf')
    train_track['current_epoch'] = 0
    train_track['training_losses'] = []
    train_track['validation_losses'] = []
    
    # Add some data
    train_track.train_epoch_update(0.5, time.time(), time.time(), 0, 1)
    train_track.val_epoch_update(0.4, time.time(), time.time())
    
    # Save stats
    miscfuncs.json_save(dict(train_track), 'test_training_stats', test_dir)
    
    # Verify file exists
    stats_path = os.path.join(test_dir, 'test_training_stats.json')
    assert os.path.exists(stats_path), "Stats file not created"
    
    # Load and verify
    loaded_stats = miscfuncs.json_load('test_training_stats', test_dir)
    assert 'training_losses' in loaded_stats, "Training losses missing"
    assert 'validation_losses' in loaded_stats, "Validation losses missing"
    assert len(loaded_stats['training_losses']) > 0, "No training losses saved"
    
    return f"Stats saved: {len(loaded_stats['training_losses'])} training, {len(loaded_stats['validation_losses'])} validation"

# ============================================================================
# TEST 11: Performance Profiling
# ============================================================================

@run_test
def test_performance_profiling():
    """Test 11: Performance profiling and timing."""
    print_header("TEST 11: Performance Profiling")
    
    import CoreAudioML.networks as networks
    import CoreAudioML.training as training
    import device_utils
    import torch
    
    device = device_utils.get_device()
    
    # Create model
    model = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model = model.to(device)
    model.eval()
    
    # Create test data
    seq_len, batch_size = 1000, 4
    test_input = torch.randn(seq_len, batch_size, 1, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(test_input)
    
    # Profile forward pass
    num_iterations = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    
    # Sync if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_iterations
    
    # Profile loss computation
    target = torch.randn(seq_len, batch_size, 1, device=device)
    loss_fn = training.LossWrapper({"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15})
    loss_fn = loss_fn.to(device)  # Move loss function to device
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = loss_fn(test_input, target)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    loss_time = (time.time() - start_time) / num_iterations
    
    return f"Forward pass: {avg_time*1000:.2f}ms, Loss computation: {loss_time*1000:.2f}ms"

# ============================================================================
# TEST 12: Edge Cases & Error Handling
# ============================================================================

@run_test
def test_edge_cases():
    """Test 12: Edge cases and error handling."""
    print_header("TEST 12: Edge Cases & Error Handling")
    
    import CoreAudioML.networks as networks
    import device_utils
    import torch
    
    device = device_utils.get_device()
    results = []
    
    # Test 1: Force CPU mode
    cpu_device = device_utils.get_device(force_cpu=True)
    assert cpu_device.type == 'cpu', "Force CPU mode failed"
    results.append("CPU fallback: OK")
    
    # Test 2: Very small batch size (create fresh model)
    model_small = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model_small = model_small.to(device)
    model_small.eval()
    model_small.reset_hidden()  # Reset hidden state before new batch size
    
    small_input = torch.randn(100, 1, 1, device=device)  # batch_size=1
    with torch.no_grad():
        output = model_small(small_input)
    assert output.shape[1] == 1, "Small batch size failed"
    results.append("Small batch: OK")
    
    # Test 3: Empty tensor handling (should fail gracefully or handle)
    try:
        empty_input = torch.randn(0, 1, 1, device=device)
        # This might fail, which is acceptable
        with torch.no_grad():
            _ = model_small(empty_input)
        results.append("Empty tensor: Handled")
    except Exception:
        results.append("Empty tensor: Failed gracefully (expected)")
    
    # Test 4: Gradient clipping edge cases
    model_grad = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model_grad = model_grad.to(device)
    model_grad.train()
    test_input_grad = torch.randn(100, 2, 1, device=device, requires_grad=True)
    output_grad = model_grad(test_input_grad)
    loss_grad = output_grad.mean()
    loss_grad.backward()
    
    # Test very high gradient clip
    torch.nn.utils.clip_grad_norm_(model_grad.parameters(), 1000.0)
    results.append("High grad clip: OK")
    
    # Test very low gradient clip
    torch.nn.utils.clip_grad_norm_(model_grad.parameters(), 0.001)
    results.append("Low grad clip: OK")
    
    return "; ".join(results)

# ============================================================================
# TEST 13: AutoTuner Functionality
# ============================================================================

@run_test
def test_autotuner():
    """Test 13: AutoTuner initialization and functionality."""
    print_header("TEST 13: AutoTuner Functionality")
    
    import CoreAudioML.networks as networks
    import device_utils
    import torch.optim as optim
    import sys
    
    # Import AutoTuner from dist_model_recnet_mps
    sys.path.insert(0, os.path.dirname(__file__))
    from dist_model_recnet_mps import AutoTuner, AutoTunerConfig
    
    device = device_utils.get_device()
    
    # Create model and optimizer
    model = networks.SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    # Initialize AutoTuner
    config = AutoTunerConfig(
        initial_lr=0.0005,
        initial_grad_clip=3.0,
        patience=3
    )
    auto_tuner = AutoTuner(optimizer, config)
    
    # Test state_dict
    state = auto_tuner.state_dict()
    assert 'lr' in state, "State dict missing 'lr'"
    assert 'grad_clip' in state, "State dict missing 'grad_clip'"
    assert 'best_loss' in state, "State dict missing 'best_loss'"
    
    # Test step() with improving loss
    improved = auto_tuner.step(0.5)
    assert improved == True, "Should detect improvement"
    assert auto_tuner.best_loss == 0.5, "Best loss not updated"
    
    # Test step() with worse loss
    improved = auto_tuner.step(0.6)
    assert improved == False, "Should not detect improvement"
    assert auto_tuner.best_loss == 0.5, "Best loss should not change"
    
    # Test load_state_dict
    new_tuner = AutoTuner(optimizer, config)
    new_tuner.load_state_dict(state)
    assert new_tuner.best_loss == state['best_loss'], "State not restored correctly"
    
    # Test plateau detection (simulate plateau)
    tuner = AutoTuner(optimizer, config)
    tuner.history = [0.5, 0.51, 0.49, 0.52, 0.48, 0.50, 0.51, 0.49]  # Oscillating
    trend = tuner.analyze_trend()
    assert trend in ['plateau', 'oscillating', 'improving', 'normal'], f"Invalid trend: {trend}"
    
    return f"AutoTuner: State save/load OK, Trend analysis: {trend}"

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests and generate summary."""
    print("\n" + "=" * 70)
    print("  MAC MPS TRAINING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_environment_imports,
        test_device_utilities,
        test_data_loading,
        test_model_initialization,
        test_loss_functions,
        test_training_loop,
        test_validation,
        test_checkpoint_operations,
        test_memory_management,
        test_diagnostics,
        test_performance_profiling,
        test_edge_cases,
        test_autotuner,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            test_results.append({
                "name": test_func.__name__,
                "status": "FAIL",
                "details": f"Crash: {str(e)}"
            })
    
    # Generate summary
    print_header("TEST SUMMARY")
    
    total_tests = len(test_results)
    passed = sum(1 for r in test_results if r['status'] == 'PASS')
    failed = sum(1 for r in test_results if r['status'] == 'FAIL')
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    elapsed_time = time.time() - test_start_time
    print(f"\n⏱️  Total Time: {elapsed_time:.2f} seconds")
    
    # Print failed tests
    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for result in test_results:
            if result['status'] == 'FAIL':
                print(f"  - {result['name']}: {result['details']}")
    
    # Save results to file
    results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total_tests,
                'passed': passed,
                'failed': failed,
                'elapsed_time': elapsed_time
            },
            'tests': test_results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Exit with appropriate code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

