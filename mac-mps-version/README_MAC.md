# Automated Guitar Amp Modelling - Mac MPS Version

This directory contains a Mac-optimized version of the guitar amplifier modelling training code with full support for Apple Silicon GPUs (MPS - Metal Performance Shaders).

## Overview

This version provides:
- **MPS Support**: Automatic detection and use of Apple Silicon GPU
- **Cross-platform compatibility**: Falls back to CUDA or CPU if MPS unavailable
- **Same interface**: Compatible with original training scripts and configs
- **Optimized for Mac**: Batch sizes and settings tuned for Mac hardware

## Requirements

### Hardware
- Apple Silicon Mac (M1, M2, M3, or newer)
- macOS 12.3 or later
- Minimum 16GB RAM (32GB+ recommended for larger models)

### Software
- Python 3.8 or later
- PyTorch 2.0 or later (with MPS support)
- All dependencies listed in `requirements.txt`

## Quick Start

1. **Navigate to this directory**:
   ```bash
   cd mac-mps-version
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Place your WAV files in `Data/train/`, `Data/val/`, and `Data/test/`
   - Files should be named `dls2-input.wav` and `dls2-target.wav` in each directory

4. **Run training**:
   ```bash
   python dist_model_recnet_mps.py --load_config RNN3 --epochs 100 --device dls2 --cuda 1
   ```

   Or use the Jupyter notebook:
   ```bash
   jupyter notebook GuitarAmp_Training_Mac.ipynb
   ```

## Key Differences from Original Version

### Device Detection
- **Original**: Only CUDA or CPU
- **Mac Version**: MPS → CUDA → CPU (automatic fallback)

### Mixed Precision
- **Original**: Uses `torch.cuda.amp.autocast()` for CUDA
- **Mac Version**: Only uses AMP on CUDA (MPS doesn't support it yet)

### Memory Management
- **Original**: CUDA-specific memory functions
- **Mac Version**: Unified interface via `device_utils.py`

### Batch Size Recommendations
- **Original**: 512+ for CUDA GPUs
- **Mac Version**: 256 recommended (can go higher with more RAM)

## Performance Comparison

### Training Speed (100 epochs, hidden_size=96, batch_size=512)

| Platform | Estimated Time | Notes |
|----------|---------------|-------|
| **T4 GPU (Colab)** | ~15-30 min | Fastest option |
| **M3 Max MPS** | ~45-90 min | Good for local development |
| **M3 Max CPU** | ~2-4 hours | Fallback if MPS unavailable |
| **M1/M2 MPS** | ~60-120 min | Slower than M3 but still faster than CPU |

### Why is Mac Slower?

1. **No Mixed Precision**: MPS doesn't support FP16 training (yet)
2. **Unified Memory**: GPU shares memory with CPU, limiting batch sizes
3. **Less Optimized**: CUDA has years of optimization, MPS is newer
4. **Architecture**: Apple GPUs optimized for different workloads

## File Structure

```
mac-mps-version/
├── CoreAudioML/          # Core training modules (copied from original)
├── Configs/              # Configuration files (same as original)
├── device_utils.py       # Cross-platform device management (NEW)
├── dist_model_recnet_mps.py  # Modified training script (MPS support)
├── GuitarAmp_Training_Mac.ipynb  # Mac-specific notebook (NEW)
├── README_MAC.md         # This file
└── requirements.txt      # Dependencies
```

## Usage Examples

### Basic Training
```bash
python dist_model_recnet_mps.py --load_config RNN3 --epochs 100 --device dls2
```

### Custom Batch Size
```bash
python dist_model_recnet_mps.py --load_config RNN3 --epochs 100 --batch_size 128 --device dls2
```

### Force CPU (if MPS causes issues)
```bash
python dist_model_recnet_mps.py --load_config RNN3 --epochs 100 --device dls2 --cuda 0
```

### Larger Model
```bash
python dist_model_recnet_mps.py --load_config RNN3 --epochs 100 --hidden_size 128 --device dls2
```

## Monitoring Training

### Activity Monitor
1. Open Activity Monitor (Applications → Utilities)
2. Window → GPU History
3. Watch GPU utilization during training

### Memory Usage
- MPS uses unified memory (shared with system)
- Monitor Activity Monitor → Memory tab
- Reduce batch_size if you see memory pressure

## Troubleshooting

### MPS Not Available
- **Check macOS version**: Requires 12.3+
- **Check PyTorch version**: Requires 2.0+
- **Verify**: Run `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory Errors
- Reduce `batch_size` in config (try 128 or 64)
- Close other applications
- Use `--cuda 0` to force CPU (slower but more memory available)

### Slow Training
- This is normal - Mac MPS is slower than NVIDIA GPUs
- Consider using Google Colab for faster training
- Verify MPS is being used: Check Activity Monitor GPU History

### Import Errors
- Make sure you're in the `mac-mps-version/` directory
- Verify `device_utils.py` exists
- Check Python path includes current directory

## Configuration Tips

### Optimal Settings for Mac M3 Max (36GB)
```json
{
  "hidden_size": 96,
  "batch_size": 256,
  "epochs": 100
}
```

### Conservative Settings (if memory issues)
```json
{
  "hidden_size": 64,
  "batch_size": 128,
  "epochs": 100
}
```

### Aggressive Settings (if you have plenty of RAM)
```json
{
  "hidden_size": 128,
  "batch_size": 512,
  "epochs": 100
}
```

## When to Use Mac vs Colab

### Use Mac Version When:
- ✅ You want to train locally without timeouts
- ✅ You need to debug or develop
- ✅ You have time and don't need fastest training
- ✅ You want full control over the environment

### Use Colab Version When:
- ✅ Speed is critical
- ✅ You need fastest training possible
- ✅ You don't mind session timeouts
- ✅ You want free GPU access

## Technical Details

### Device Detection Priority
1. **MPS** (Apple Silicon GPU) - if available
2. **CUDA** (NVIDIA GPU) - if available
3. **CPU** - fallback

### Memory Management
- MPS uses unified memory architecture
- No separate VRAM - shares system RAM
- Can use larger batch sizes if you have more RAM
- Monitor via Activity Monitor, not nvidia-smi

### Mixed Precision
- Not available on MPS (CUDA-only feature)
- Training runs in FP32 (full precision)
- This is normal and expected

## Compatibility

- ✅ Same config files as original version
- ✅ Same command-line interface
- ✅ Same output format and structure
- ✅ Models trained on Mac can be used with original version
- ✅ Can fall back to CUDA if available (e.g., external GPU)

## Support

For issues specific to Mac/MPS:
1. Check PyTorch MPS documentation
2. Verify macOS and PyTorch versions
3. Try forcing CPU mode (`--cuda 0`) to isolate MPS issues

For general training issues, refer to the main project README.

## Performance Benchmarks

Based on typical training runs (hidden_size=96, batch_size=256):

| Metric | M3 Max MPS | T4 GPU (Colab) | Ratio |
|--------|------------|----------------|-------|
| Time per epoch | ~30-60 sec | ~10-20 sec | 2-3x slower |
| Memory usage | ~8-12 GB | ~4-6 GB | Higher (unified) |
| Batch size max | 512+ | 1024+ | Lower on Mac |

*Note: Actual performance varies based on model size, batch size, and data complexity.*

## Future Improvements

- MPS mixed precision support (when PyTorch adds it)
- Better memory management for unified memory
- Optimized batch sizes for different Mac models
- Native MPS profiling support






