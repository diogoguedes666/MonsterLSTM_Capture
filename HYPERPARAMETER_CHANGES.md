# Hyperparameter Changes - DAFx'19 Alignment and HF Hinge Loss

## Overview

This document tracks hyperparameter changes made to align with DAFx'19 paper recommendations and implement the high-frequency hinge loss for better control of high-frequency artifacts in distortion pedal modeling.

## DAFx'19 Paper Reference

Based on "Real-Time Black-Box Modelling with Recurrent Neural Networks" (Wright et al., DAFx'19):
- Paper: https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf
- Key recommendations: LSTM units, hidden_size 64-96, Adam optimizer with lr=5e-4, gradient clipping 3-5

## Changes Made

### 1. High-Frequency Hinge Loss Implementation (`CoreAudioML/training.py`)

**New Loss Function: `HFHingeLoss`**
- **Purpose**: Asymmetrically penalizes excess high frequencies (only when model output > target)
- **Key Features**:
  - Only penalizes excess, not deficit (prevents dulling legitimate high-frequency content)
  - Uses `torch.relu()` to isolate excess frequencies
  - Applies `torch.log1p()` for smooth penalty scaling
  - Configurable frequency threshold (`fmin`, default: 10kHz) and strength multiplier
- **Parameters**:
  - `n_fft=2048`: STFT window size
  - `hop_length=None`: Defaults to n_fft//4
  - `sample_rate=44100`: Audio sample rate
  - `fmin=10000`: Minimum frequency threshold (Hz) for HF penalty
  - `strength=0.5`: Multiplier for the loss term

**Deprecated: `SpectralLoss`**
- Marked as deprecated but kept for backward compatibility
- Shows deprecation warning when used
- Users should migrate to `HFHingeLoss` for better results

### 2. Optimizer Defaults Updated (`dist_model_recnet.py`)

**Learning Rate**:
- **Old**: `0.005` (5e-3)
- **New**: `0.0005` (5e-4) - DAFx'19 recommendation

**Gradient Clipping**:
- **Old**: `1.0`
- **New**: `3.0` - DAFx'19 safe zone (3-5)

**Optimizer Type**:
- **Old**: `torch.optim.Adam` with `betas=(0.9, 0.99)`
- **New**: `torch.optim.AdamW` with `betas=(0.9, 0.999)` - Standard Adam defaults, DAFx'19 compatible

**Weight Decay Special Handling**:
- **Non-recurrent weights**: Use default `weight_decay=1e-4`
- **Recurrent weights** (`weight_hh*`): Use lower `weight_decay=3e-5` to preserve LSTM memory capacity and bloom

### 3. Hyperparameter Defaults Updated

**Hidden Size**:
- **Old**: `32`
- **New**: `96` - DAFx'19 recommendation (64-96 for real-time, 96 is good middle ground)

**Loss Function Defaults**:
- **Old**: `{'ESRPre': 0.70, 'DC': 0.10, 'Spectral': 0.20}`
- **New**: `{'ESR': 0.50, 'DC': 0.10, 'HFHinge': 0.10}`
- **Note**: Config files can override these defaults

### 4. Config File Support (`Configs/RNN3.json`)

**Updated Defaults**:
- `hidden_size: 96` (from 64)
- `loss_fcns: {"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15}` (replaces Spectral with HFHinge, restores ESR to 0.75 as per original)

**New Configurable Parameters**:
- `hf_hinge_fmin`: Minimum frequency threshold for HF hinge loss (default: 10000 Hz)
- `hf_hinge_strength`: Strength multiplier for HF hinge loss (default: 0.5)

**Config File Override Support**:
All new parameters can be overridden via JSON config files:
- `learn_rate`
- `gradient_clip`
- `weight_decay`
- `hidden_size`
- `loss_fcns`
- `hf_hinge_fmin`
- `hf_hinge_strength`

## Expected Impact

### HF Hinge Loss Benefits:
1. **Asymmetric Penalty**: Only penalizes excess high frequencies, preventing dulling of legitimate content
2. **Better Artifact Control**: Specifically targets the 10kHz+ range where artifacts are most problematic
3. **Smooth Scaling**: `log1p()` provides smooth penalty scaling that doesn't overwhelm the loss function

### DAFx'19 Alignment Benefits:
1. **Stable Training**: Lower learning rate (5e-4) provides more stable convergence
2. **Better Gradient Control**: Gradient clipping at 3.0 prevents explosions while allowing tone learning
3. **Preserved Memory**: Lower weight decay on recurrent weights maintains LSTM memory capacity
4. **Optimal Model Size**: Hidden size of 96 balances accuracy and real-time performance

## Migration Guide

### From Spectral Loss to HF Hinge Loss:

**Old Config**:
```json
{
  "loss_fcns": {"ESR": 0.65, "DC": 0.10, "Spectral": 0.25}
}
```

**New Config**:
```json
{
  "loss_fcns": {"ESR": 0.75, "DC": 0.10, "HFHinge": 0.15},
  "hf_hinge_fmin": 10000,
  "hf_hinge_strength": 0.5
}
```

### Backward Compatibility:

- Old configs using `Spectral` loss will still work but show a deprecation warning
- All existing hyperparameters remain configurable via config files
- Default values are updated but can be overridden

## Testing

After updating hyperparameters:

1. **Restart training** with the updated config:
   ```bash
   python dist_model_recnet.py --load_config RNN3
   ```

2. **Monitor the frequency response** during training using the diagnostics dashboard

3. **Compare results** - expect:
   - Reduced excess energy above 10kHz
   - Better amplitude matching (due to restored ESR=0.75)
   - More stable training (due to DAFx'19 optimizer settings)

## Notes

- **HF Hinge Loss** is asymmetric - only penalizes when model output > target in high frequencies
- This prevents dulling legitimate high-frequency content while suppressing artifacts
- **DAFx'19 paper** used ESR as primary loss, which aligns with keeping ESR at 0.75
- **Lower weight decay** on recurrent weights preserves LSTM memory capacity and "bloom" characteristics
- **Hidden size of 96** balances accuracy and real-time performance per DAFx'19 recommendations

## References

- Wright, A., Damskägg, E.-P., & Välimäki, V. (2019). "Real-Time Black-Box Modelling with Recurrent Neural Networks." DAFx'19.
- Paper: https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf
