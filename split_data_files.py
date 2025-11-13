#!/usr/bin/env python3
"""
Split all audio files in the Data directory in half
"""

import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path


def split_audio_file(filepath, output_dir=None):
    """
    Split an audio file in half and save both halves
    
    Args:
        filepath: Path to the input audio file
        output_dir: Directory to save split files (default: same as input)
    """
    # Read the audio file
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Convert to float if needed
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    # Handle mono/stereo
    if len(audio_data.shape) > 1:
        # Stereo or multi-channel
        total_samples = audio_data.shape[0]
        half_point = total_samples // 2
        
        first_half = audio_data[:half_point]
        second_half = audio_data[half_point:]
    else:
        # Mono
        total_samples = len(audio_data)
        half_point = total_samples // 2
        
        first_half = audio_data[:half_point]
        second_half = audio_data[half_point:]
    
    # Get file info
    file_path = Path(filepath)
    base_name = file_path.stem
    parent_dir = file_path.parent if output_dir is None else Path(output_dir)
    
    # Convert back to int16 for saving
    def to_int16(data):
        if data.dtype == np.float32:
            return (data * 32767).astype(np.int16)
        return data
    
    # Save first half
    first_half_path = parent_dir / f"{base_name}_part1.wav"
    wavfile.write(str(first_half_path), sample_rate, to_int16(first_half))
    print(f"Saved first half: {first_half_path} ({len(first_half)/sample_rate:.2f}s)")
    
    # Save second half
    second_half_path = parent_dir / f"{base_name}_part2.wav"
    wavfile.write(str(second_half_path), sample_rate, to_int16(second_half))
    print(f"Saved second half: {second_half_path} ({len(second_half)/sample_rate:.2f}s)")
    
    return first_half_path, second_half_path


def split_all_files_in_directory(data_dir="Data"):
    """
    Split all WAV files in train/val/test directories
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
    """
    data_path = Path(data_dir)
    
    # Process each subdirectory
    for subdir in ['train', 'val', 'test']:
        subdir_path = data_path / subdir
        if not subdir_path.exists():
            print(f"Warning: {subdir_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {subdir} directory...")
        
        # Find all WAV files
        wav_files = list(subdir_path.glob("*.wav"))
        
        if not wav_files:
            print(f"No WAV files found in {subdir}")
            continue
        
        for wav_file in wav_files:
            print(f"\nSplitting {wav_file.name}...")
            try:
                split_audio_file(wav_file, output_dir=subdir_path)
            except Exception as e:
                print(f"Error splitting {wav_file}: {e}")
    
    print("\n" + "="*50)
    print("All files have been split!")


if __name__ == "__main__":
    split_all_files_in_directory("Data")

