"""
Test script to compare frequency response between model output and target.
This helps diagnose high-frequency issues in the trained model.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys

def analyze_frequency_response(model_path, target_path, output_dir=None):
    """
    Analyze frequency response differences between model output and target.
    
    Args:
        model_path: Path to model output WAV file
        target_path: Path to target WAV file
        output_dir: Optional directory to save plots
    """
    # Load audio files
    model_sr, model_audio = wavfile.read(model_path)
    target_sr, target_audio = wavfile.read(target_path)
    
    # Convert to float32
    if model_audio.dtype == np.int16:
        model_audio = model_audio.astype(np.float32) / 32768.0
    elif model_audio.dtype == np.int32:
        model_audio = model_audio.astype(np.float32) / 2147483648.0
    
    if target_audio.dtype == np.int16:
        target_audio = target_audio.astype(np.float32) / 32768.0
    elif target_audio.dtype == np.int32:
        target_audio = target_audio.astype(np.float32) / 2147483648.0
    
    # Handle stereo
    if len(model_audio.shape) > 1:
        model_audio = np.mean(model_audio, axis=1)
    if len(target_audio.shape) > 1:
        target_audio = np.mean(target_audio, axis=1)
    
    # Ensure same length
    min_len = min(len(model_audio), len(target_audio))
    model_audio = model_audio[:min_len]
    target_audio = target_audio[:min_len]
    
    # Compute frequency response using Welch method
    freqs, target_psd = signal.welch(target_audio, fs=target_sr, nperseg=2048, 
                                     window='hann', noverlap=1024, average='mean')
    _, model_psd = signal.welch(model_audio, fs=model_sr, nperseg=2048, 
                               window='hann', noverlap=1024, average='mean')
    
    # Convert to dB
    target_db = 10 * np.log10(target_psd + 1e-12)
    model_db = 10 * np.log10(model_psd + 1e-12)
    
    # Calculate difference
    diff_db = model_db - target_db
    
    # Analyze specific frequency bands
    mask_10k = freqs >= 10000
    mask_16k = freqs >= 16000
    mask_5k_10k = (freqs >= 5000) & (freqs < 10000)
    
    print("=" * 60)
    print("FREQUENCY RESPONSE ANALYSIS")
    print("=" * 60)
    print(f"\nTarget file: {target_path}")
    print(f"Model file: {model_path}")
    print(f"Sample rate: {target_sr} Hz")
    print(f"Length: {min_len / target_sr:.2f} seconds")
    
    print("\n--- Frequency Band Analysis ---")
    print(f"5-10kHz band:")
    print(f"  Target avg: {np.mean(target_db[mask_5k_10k]):.2f} dB")
    print(f"  Model avg:  {np.mean(model_db[mask_5k_10k]):.2f} dB")
    print(f"  Difference: {np.mean(diff_db[mask_5k_10k]):.2f} dB")
    
    print(f"\n10kHz+ band:")
    print(f"  Target avg: {np.mean(target_db[mask_10k]):.2f} dB")
    print(f"  Model avg:  {np.mean(model_db[mask_10k]):.2f} dB")
    print(f"  Difference: {np.mean(diff_db[mask_10k]):.2f} dB")
    
    print(f"\n16kHz+ band:")
    print(f"  Target avg: {np.mean(target_db[mask_16k]):.2f} dB")
    print(f"  Model avg:  {np.mean(model_db[mask_16k]):.2f} dB")
    print(f"  Difference: {np.mean(diff_db[mask_16k]):.2f} dB")
    
    # Find peak excess
    excess_10k = diff_db[mask_10k]
    if len(excess_10k) > 0:
        max_excess_idx = np.argmax(excess_10k)
        max_excess_freq = freqs[mask_10k][max_excess_idx]
        max_excess_db = excess_10k[max_excess_idx]
        print(f"\nPeak excess above 10kHz: {max_excess_db:.2f} dB at {max_excess_freq:.1f} Hz")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Frequency response comparison
    ax1 = axes[0]
    ax1.semilogx(freqs, target_db, label='Target', color='#007AFF', linewidth=2, alpha=0.8)
    ax1.semilogx(freqs, model_db, label='Model', color='#AF52DE', linewidth=2, alpha=0.8)
    ax1.axvline(x=10000, color='red', linestyle='--', alpha=0.5, label='10kHz')
    ax1.axvline(x=16000, color='orange', linestyle='--', alpha=0.5, label='16kHz')
    ax1.set_xlim([20, 22000])
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title('Frequency Response Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference (Model - Target)
    ax2 = axes[1]
    ax2.semilogx(freqs, diff_db, color='#FF3B30', linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=10000, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=16000, color='orange', linestyle='--', alpha=0.5)
    ax2.fill_between(freqs, 0, diff_db, where=(diff_db > 0), 
                    color='#FF3B30', alpha=0.2, label='Excess')
    ax2.fill_between(freqs, 0, diff_db, where=(diff_db < 0), 
                    color='#34C759', alpha=0.2, label='Deficit')
    ax2.set_xlim([20, 22000])
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Difference (dB)', fontsize=12)
    ax2.set_title('Frequency Difference (Model - Target)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'frequency_response_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'freqs': freqs,
        'target_db': target_db,
        'model_db': model_db,
        'diff_db': diff_db,
        'excess_10k_db': np.mean(diff_db[mask_10k]),
        'excess_16k_db': np.mean(diff_db[mask_16k])
    }

if __name__ == '__main__':
    # Default paths
    results_dir = 'Results/dls2-RNN3'
    data_dir = 'Data/val'
    
    model_path = os.path.join(results_dir, 'best_val_out.wav')
    target_files = [f for f in os.listdir(data_dir) if 'target' in f and f.endswith('.wav')]
    
    if not target_files:
        print(f"Error: No target file found in {data_dir}")
        sys.exit(1)
    
    target_path = os.path.join(data_dir, target_files[0])
    
    if not os.path.exists(model_path):
        print(f"Error: Model output not found at {model_path}")
        sys.exit(1)
    
    analyze_frequency_response(model_path, target_path, output_dir=results_dir)

