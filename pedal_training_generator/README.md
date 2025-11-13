# Pedal Training File Generator

This directory contains a comprehensive Python script for generating training files optimized for LSTM pedal capture.

## Overview

The `generate_pedal_training_files.py` script generates at least 30 different types of spectral sounds designed to capture the full range of characteristics that guitar pedals exhibit. These signals include:

- **Frequency Sweeps**: Logarithmic, linear, exponential, and multi-band sweeps
- **Amplitude Variations**: Linear and logarithmic amplitude sweeps, amplitude modulation
- **Noise Types**: White, pink, brown, and band-limited noise
- **Harmonic Content**: Harmonic series, inharmonic series, power chords, complex chords
- **Guitar Techniques**: Vibrato, palm muting, pinch harmonics, sliding notes, tremolo picking
- **Advanced Signals**: Phase sweeps, intermodulation distortion tests, crest factor variations, transient response tests, nonlinear tests, time-varying spectra

## Requirements

- Python 3.7+
- numpy
- scipy
- Standard library modules (os, json, datetime, pathlib)

## Usage

### Basic Usage

```bash
python generate_pedal_training_files.py
```

This will generate training files with default settings:
- Sample rate: 44100 Hz
- Base duration: 10 seconds per signal type
- Output directory: `pedal_training_data/`
- Filename prefix: `pedal`
- Train/Val/Test split: 70/15/15

### Advanced Usage

```bash
python generate_pedal_training_files.py \
    --sample_rate 44100 \
    --duration 15.0 \
    --output_dir my_training_data \
    --filename_prefix mypedal \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --variations 5
```

### Command Line Arguments

- `--sample_rate`: Audio sample rate in Hz (default: 44100)
- `--duration`: Base duration per signal type in seconds (default: 10.0)
- `--output_dir`: Output directory name (default: pedal_training_data)
- `--filename_prefix`: Prefix for output filenames (default: pedal)
- `--train_ratio`: Training set ratio (default: 0.7)
- `--val_ratio`: Validation set ratio (default: 0.15)
- `--test_ratio`: Test set ratio (default: 0.15)
- `--variations`: Number of variations per signal type (default: 3)

## Output Structure

The script generates the following directory structure:

```
pedal_training_data/
├── train/
│   ├── pedal-input.wav
│   └── pedal-target.wav
├── val/
│   ├── pedal-input.wav
│   └── pedal-target.wav
├── test/
│   ├── pedal-input.wav
│   └── pedal-target.wav
└── metadata.json
```

## Signal Types Generated

The script generates 33+ different signal types:

1. Logarithmic frequency sweep (20Hz - 20kHz)
2. Linear frequency sweep (20Hz - 20kHz)
3. Exponential frequency sweep (20Hz - 20kHz)
4. Multi-band frequency sweeps (sub-bass, bass, mid, high)
5. Linear amplitude sweep (0.1 to 1.0)
6. Logarithmic amplitude sweep (0.01 to 1.0)
7. Chirp with amplitude modulation
8. White noise bursts
9. Pink noise (1/f noise)
10. Brown noise (1/f² noise)
11. Band-limited noise (sub-bass)
12. Band-limited noise (mid)
13. Band-limited noise (high)
14. Multi-band noise bursts
15. Impulse train
16. Harmonic series (15 harmonics)
17. Inharmonic series (stretched harmonics)
18. Power chords
19. Complex chords (jazz/extended)
20. Vibrato (single note with vibrato)
21. Palm-muted signals
22. Pinch harmonics
23. Sliding notes (frequency glides)
24. Tremolo picking
25. Dynamic playing (varying attack/release)
26. Phase sweep (0 to 2π)
27. Intermodulation distortion test (two-tone IMD)
28. Crest factor variations
29. Transient response test
30. Nonlinear test signals
31. Time-varying spectra (LFO-modulated)
32. Combined signals (multiple techniques)
33. Real-world guitar patterns

Each signal type is generated with multiple amplitude levels (0.01, 0.1, 0.3, 0.5, 0.7, 1.0) to capture nonlinear behavior at different input levels.

## Integration with Training Script

The generated files follow the naming convention expected by `dist_model_recnet.py`:

- Input files: `{prefix}-input.wav`
- Target files: `{prefix}-target.wav`

To use with the training script:

```bash
python dist_model_recnet.py \
    --data_location pedal_training_data \
    --file_name pedal \
    --device pedal
```

## Metadata

The script generates a `metadata.json` file containing:
- Sample rate
- Base duration
- Generation date
- List of all signal types generated
- Number of variations per type
- Amplitude levels used

## Notes

- All signals are normalized to [-1, 1] range before saving
- Signals are saved as 16-bit WAV files
- The target files are initially identical to input files (meant to be processed through the pedal)
- The script generates multiple variations of each signal type for robust training

