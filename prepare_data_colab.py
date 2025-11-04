#!/usr/bin/env python3
"""
Data preparation script for Google Colab training.
This script helps prepare and validate audio data for guitar amp modelling training.
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
from pathlib import Path

def validate_audio_file(filepath, expected_sample_rate=44100):
    """Validate an audio file for training compatibility."""
    try:
        sample_rate, data = wavfile.read(filepath)

        # Check sample rate
        if sample_rate != expected_sample_rate:
            print(f"⚠️  Warning: {filepath} has sample rate {sample_rate}Hz, expected {expected_sample_rate}Hz")
            return False

        # Check if mono or stereo
        if len(data.shape) == 1:
            channels = 1
            length_samples = len(data)
        else:
            channels, length_samples = data.shape

        # Convert to float for analysis
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            if data.dtype == np.int16:
                data = data / 32768.0
            elif data.dtype == np.int32:
                data = data / 2147483648.0

        # Basic audio analysis
        duration = length_samples / sample_rate
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))

        print(f"📊 Duration: {duration:.1f}s, Channels: {channels}, RMS: {rms:.3f}, Peak: {peak:.3f}")
        # Check for DC offset
        dc_offset = np.mean(data)
        if abs(dc_offset) > 0.01:
            print(f"⚠️  DC offset detected: {dc_offset:.3f}")
        return True

    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

def prepare_data_directories():
    """Create necessary data directories."""
    dirs = ['Data/train', 'Data/val', 'Data/test', 'Results', 'Configs']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def generate_test_data(duration_train=30, duration_val=10, duration_test=5):
    """Generate synthetic test data for development."""
    print("🎵 Generating synthetic test data...")

    def create_test_signal(duration, freq1=440, freq2=880, freq3=1320):
        """Create a test signal with multiple harmonics."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Mix of harmonics to simulate guitar-like content
        signal = (0.4 * np.sin(2 * np.pi * freq1 * t) +  # Fundamental
                 0.3 * np.sin(2 * np.pi * freq2 * t) +  # 2nd harmonic
                 0.2 * np.sin(2 * np.pi * freq3 * t) +  # 3rd harmonic
                 0.1 * np.sin(2 * np.pi * freq1 * 2 * t))  # Octave

        # Normalize and add some noise
        signal = signal / np.max(np.abs(signal)) * 0.8
        noise = np.random.normal(0, 0.01, len(signal))
        signal += noise

        return (signal * 32767).astype(np.int16), sample_rate

    # Generate input signals (clean guitar-like)
    input_train, sr = create_test_signal(duration_train)
    input_val, _ = create_test_signal(duration_val)
    input_test, _ = create_test_signal(duration_test)

    # Generate target signals (simulate amp processing - add some distortion/processing)
    def process_target(signal):
        """Simple amp simulation: add soft clipping and compression."""
        # Soft clipping
        target = np.tanh(signal * 1.5) * 0.9
        # Add some harmonic enhancement
        target = target + 0.1 * np.sin(2 * np.pi * 1320 * np.linspace(0, len(signal)/sr, len(signal), False))
        return (target / np.max(np.abs(target)) * 32767 * 0.8).astype(np.int16)

    target_train = process_target(input_train.astype(np.float32) / 32767)
    target_val = process_target(input_val.astype(np.float32) / 32767)
    target_test = process_target(input_test.astype(np.float32) / 32767)

    # Save files
    files_to_create = [
        ('Data/train/dls2-input.wav', input_train, sr),
        ('Data/train/dls2-target.wav', target_train, sr),
        ('Data/val/dls2-input.wav', input_val, sr),
        ('Data/val/dls2-target.wav', target_val, sr),
        ('Data/test/dls2-input.wav', input_test, sr),
        ('Data/test/dls2-target.wav', target_test, sr),
    ]

    for filepath, data, sample_rate in files_to_create:
        wavfile.write(filepath, sample_rate, data)
        print(f"✓ Generated: {filepath}")

    print("✓ Synthetic data generation complete!")
    print("Note: This is for testing only. Real guitar recordings will give much better results.")

def validate_all_data():
    """Validate all training data files."""
    print("🔍 Validating training data...")

    required_files = [
        'Data/train/dls2-input.wav',
        'Data/train/dls2-target.wav',
        'Data/val/dls2-input.wav',
        'Data/val/dls2-target.wav',
        'Data/test/dls2-input.wav',
        'Data/test/dls2-target.wav'
    ]

    all_valid = True
    for filepath in required_files:
        if os.path.exists(filepath):
            if not validate_audio_file(filepath):
                all_valid = False
        else:
            print(f"✗ Missing file: {filepath}")
            all_valid = False

    if all_valid:
        print("✓ All data files validated successfully!")
    else:
        print("✗ Some data files have issues. Please check the warnings above.")

    return all_valid

def main():
    parser = argparse.ArgumentParser(description='Prepare audio data for guitar amp modelling training')
    parser.add_argument('--generate-test-data', action='store_true',
                       help='Generate synthetic test data instead of using real recordings')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing data files')
    parser.add_argument('--setup-directories', action='store_true',
                       help='Only create necessary directories')

    args = parser.parse_args()

    if args.setup_directories:
        prepare_data_directories()
        return

    if args.validate_only:
        validate_all_data()
        return

    if args.generate_test_data:
        prepare_data_directories()
        generate_test_data()
        validate_all_data()
    else:
        print("No action specified. Use --generate-test-data or --validate-only")
        print("For real training, upload your own WAV files to Data/train/, Data/val/, Data/test/")

if __name__ == '__main__':
    main()
