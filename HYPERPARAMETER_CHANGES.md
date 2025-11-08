# Hyperparameter Changes to Fix High-Frequency Exaggeration

## Problem Identified
The trained model shows significant excess energy above 10kHz:
- **7.31 dB average excess** above 10kHz
- **Peak excess of 15.04 dB** at ~10.3kHz
- This causes harsh, exaggerated distortion in the upper high frequencies

## Root Cause
1. The `SpectralLoss` was configured to target frequencies above **5kHz** with a penalty of **3.0**
2. The actual problem frequencies are above **10kHz**, which weren't being penalized effectively
3. The config (`RNN3.json`) didn't include `SpectralLoss` at all, only using `ESR` and `DC` losses

## Changes Made

### 1. Modified SpectralLoss Defaults (`CoreAudioML/training.py`)
   - **`high_cutoff`**: Changed from `5000` Hz → **`10000` Hz** (10kHz)
   - **`mid_cutoff`**: Changed from `3000` Hz → **`5000` Hz** (5kHz) 
   - **`excess_penalty`**: Increased from `3.0` → **`10.0`** (3.3x stronger penalty)
   - Updated comments to reflect 10kHz+ targeting

### 2. Updated Training Config (`Configs/RNN3.json`)
   - **Added SpectralLoss** to the loss function mix
   - **Loss weights**:
     - `ESR`: 0.65 (down from 0.9) - time-domain matching
     - `DC`: 0.10 (unchanged) - DC offset correction
     - `Spectral`: 0.25 (new) - frequency-domain matching, especially 10kHz+

## Expected Impact

The SpectralLoss will now:
1. **Heavily penalize** (10x penalty) when model output has MORE energy than target above 10kHz
2. **Focus on the problematic frequency range** (10kHz+) instead of 5kHz+
3. **Balance time-domain and frequency-domain losses** to prevent overfitting to high-frequency noise

## Next Steps

1. **Restart training** with the updated config:
   ```bash
   python dist_model_recnet.py --load_config RNN3
   ```

2. **Monitor the frequency response** during training using the diagnostics dashboard

3. **Compare results** after training - the excess above 10kHz should be significantly reduced

## Testing

A test script (`test_frequency_response.py`) has been created to analyze frequency response differences. Run it to verify improvements:
```bash
python test_frequency_response.py
```

## Notes

- The `excess_penalty` of 10.0 is aggressive but necessary given the 15dB peak excess observed
- The SpectralLoss weight of 0.25 balances frequency-domain constraints without overwhelming time-domain accuracy
- These changes target the specific issue (10kHz+ exaggeration) while maintaining overall model performance

