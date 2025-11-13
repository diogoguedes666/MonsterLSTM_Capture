#!/usr/bin/env python3
"""
Create a smaller version of the Data folder by trimming audio files to 1/10th their size.
This is useful for faster uploads and testing in Colab.
"""

import os
import shutil
from scipy.io import wavfile
import numpy as np

def trim_audio_file(input_path, output_path, fraction=0.1):
    """
    Trim an audio file to the first fraction (default 1/10th) of its length.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the trimmed audio file
        fraction: Fraction of the original to keep (default 0.1 for 1/10th)
    """
    # Read the audio file
    sample_rate, audio_data = wavfile.read(input_path)
    
    # Calculate the number of samples to keep (first fraction)
    total_samples = len(audio_data)
    samples_to_keep = int(total_samples * fraction)
    
    # Trim the audio data
    trimmed_audio = audio_data[:samples_to_keep]
    
    # Write the trimmed audio file
    wavfile.write(output_path, sample_rate, trimmed_audio)
    
    print(f"  Trimmed {input_path}: {total_samples} -> {samples_to_keep} samples ({fraction*100:.1f}%)")

def create_small_data_folder(source_dir='Data', dest_dir='DataSmall', fraction=0.1):
    """
    Create a duplicate of the Data folder with trimmed audio files.
    
    Args:
        source_dir: Source directory to copy from
        dest_dir: Destination directory name
        fraction: Fraction of audio files to keep (default 0.1 for 1/10th)
    """
    source_path = os.path.abspath(source_dir)
    dest_path = os.path.abspath(dest_dir)
    
    # Check if source directory exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    # Remove destination directory if it exists
    if os.path.exists(dest_path):
        print(f"Removing existing {dest_dir} directory...")
        shutil.rmtree(dest_path)
    
    # Create destination directory structure
    print(f"Creating {dest_dir} directory structure...")
    os.makedirs(dest_path, exist_ok=True)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        # Calculate relative path from source
        rel_path = os.path.relpath(root, source_path)
        
        # Create corresponding directory in destination
        if rel_path == '.':
            dest_root = dest_path
        else:
            dest_root = os.path.join(dest_path, rel_path)
        os.makedirs(dest_root, exist_ok=True)
        
        # Process each file
        for file in files:
            source_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            
            # Handle .wav files - trim them
            if file.endswith('.wav'):
                print(f"Processing: {source_file}")
                trim_audio_file(source_file, dest_file, fraction)
            # Copy other files as-is (like .asd files)
            else:
                shutil.copy2(source_file, dest_file)
                print(f"Copied: {source_file} -> {dest_file}")
    
    print(f"\nDone! Created {dest_dir} with audio files trimmed to {fraction*100:.1f}% of original size.")

if __name__ == '__main__':
    create_small_data_folder('Data', 'DataSmall', fraction=0.1)






