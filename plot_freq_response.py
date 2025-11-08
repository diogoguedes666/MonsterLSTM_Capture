#!/usr/bin/env python3
"""
Frequency Response Visualization Tool

Plots the average frequency response of target vs model output audio files
across the full frequency range for easy comparison.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os


def load_audio(filepath):
    """Load audio file and return sample rate and normalized audio data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Convert to float32 and normalize to [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    # Handle mono/stereo
    if len(audio_data.shape) > 1:
        # Convert to mono by averaging channels
        audio_data = np.mean(audio_data, axis=1)
    
    return sample_rate, audio_data


def compute_frequency_response(audio_data, sample_rate, nperseg=2048):
    """
    Compute average frequency response using Welch's method.
    
    Args:
        audio_data: Audio signal array
        sample_rate: Sample rate in Hz
        nperseg: Length of each segment for Welch's method
    
    Returns:
        frequencies: Frequency array in Hz
        magnitude_db: Magnitude in dB
    """
    # Use Welch's method to compute power spectral density
    frequencies, psd = signal.welch(
        audio_data,
        fs=sample_rate,
        nperseg=nperseg,
        window='hann',
        noverlap=nperseg // 2,
        average='mean'
    )
    
    # Convert to dB scale, avoiding log(0)
    psd = np.maximum(psd, 1e-12)  # Avoid log(0)
    magnitude_db = 10 * np.log10(psd)
    
    return frequencies, magnitude_db


def plot_frequency_response(target_path, output_path, save_path=None, show_plot=True):
    """
    Plot frequency response comparison between target and model output.
    
    Args:
        target_path: Path to target audio file
        output_path: Path to model output audio file
        save_path: Optional path to save plot (if None, shows interactively)
    """
    print(f"Loading target audio: {target_path}")
    target_sr, target_audio = load_audio(target_path)
    
    print(f"Loading model output audio: {output_path}")
    output_sr, output_audio = load_audio(output_path)
    
    # Ensure both have the same sample rate
    if target_sr != output_sr:
        print(f"Warning: Sample rates differ (target: {target_sr} Hz, output: {output_sr} Hz)")
        print(f"Using target sample rate: {target_sr} Hz")
        # Resample output to match target (simple approach - use minimum length)
        min_length = min(len(target_audio), len(output_audio))
        target_audio = target_audio[:min_length]
        output_audio = output_audio[:min_length]
        sample_rate = target_sr
    else:
        sample_rate = target_sr
        # Trim to same length
        min_length = min(len(target_audio), len(output_audio))
        target_audio = target_audio[:min_length]
        output_audio = output_audio[:min_length]
    
    print(f"Computing frequency response...")
    print(f"  Audio length: {min_length / sample_rate:.2f} seconds")
    
    # Compute frequency responses
    target_freqs, target_mag_db = compute_frequency_response(target_audio, sample_rate)
    output_freqs, output_mag_db = compute_frequency_response(output_audio, sample_rate)
    
    # Calculate statistics
    # Find frequency range indices
    freq_mask = (target_freqs >= 20) & (target_freqs <= 22000)
    target_masked = target_mag_db[freq_mask]
    output_masked = output_mag_db[freq_mask]
    freqs_masked = target_freqs[freq_mask]
    
    # RMS difference
    diff_db = output_masked - target_masked
    rms_diff = np.sqrt(np.mean(diff_db ** 2))
    
    # Max difference and frequency
    max_diff_idx = np.argmax(np.abs(diff_db))
    max_diff_db = diff_db[max_diff_idx]
    max_diff_freq = freqs_masked[max_diff_idx]
    
    # Print statistics
    print("\nFrequency Response Statistics:")
    print(f"  RMS difference: {rms_diff:.2f} dB")
    print(f"  Max difference: {max_diff_db:.2f} dB at {max_diff_freq:.1f} Hz")
    
    # Detailed frequency-by-amplitude analysis
    print("\n" + "="*80)
    print("DETAILED FREQUENCY-BY-AMPLITUDE ANALYSIS")
    print("="*80)
    print(f"{'Frequency (Hz)':<15} {'Target (dB)':<15} {'Model (dB)':<15} {'Difference (dB)':<15} {'Status':<15}")
    print("-"*80)
    
    # Sample key frequency points for analysis
    key_freqs = [20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 6000, 8000, 10000, 
                 12000, 14000, 16000, 18000, 20000, 22000]
    
    for freq in key_freqs:
        if freq < target_freqs[0] or freq > target_freqs[-1]:
            continue
        
        # Find closest frequency bin
        target_idx = np.argmin(np.abs(target_freqs - freq))
        output_idx = np.argmin(np.abs(output_freqs - freq))
        
        target_amp = target_mag_db[target_idx]
        output_amp = output_mag_db[output_idx]
        diff = output_amp - target_amp
        
        # Determine status
        if abs(diff) < 1.0:
            status = "✓ Good"
        elif abs(diff) < 2.0:
            status = "~ Fair"
        elif diff > 2.0:
            status = "↑ Excess"
        else:
            status = "↓ Deficient"
        
        print(f"{freq:<15.1f} {target_amp:<15.2f} {output_amp:<15.2f} {diff:<15.2f} {status:<15}")
    
    # Additional analysis: frequency bands
    print("\n" + "-"*80)
    print("FREQUENCY BAND ANALYSIS")
    print("-"*80)
    
    bands = [
        (20, 200, "Sub-bass"),
        (200, 2000, "Bass/Mid"),
        (2000, 5000, "Mid-high"),
        (5000, 10000, "High"),
        (10000, 16000, "Very High"),
        (16000, 22000, "Ultra High")
    ]
    
    for low_freq, high_freq, name in bands:
        band_mask = (target_freqs >= low_freq) & (target_freqs <= high_freq)
        if not np.any(band_mask):
            continue
        
        target_band = target_mag_db[band_mask]
        output_band = output_mag_db[band_mask]
        diff_band = output_band - target_band
        
        avg_target = np.mean(target_band)
        avg_output = np.mean(output_band)
        avg_diff = np.mean(diff_band)
        rms_diff_band = np.sqrt(np.mean(diff_band ** 2))
        
        print(f"\n{name} ({low_freq}-{high_freq} Hz):")
        print(f"  Avg Target: {avg_target:.2f} dB")
        print(f"  Avg Model:  {avg_output:.2f} dB")
        print(f"  Avg Diff:   {avg_diff:.2f} dB")
        print(f"  RMS Diff:   {rms_diff_band:.2f} dB")
    
    print("\n" + "="*80)
    
    # Create plot if requested
    if show_plot or save_path:
        plt.figure(figsize=(12, 6))
        
        # Plot both traces
        plt.semilogx(target_freqs, target_mag_db, label='Target', linewidth=1.5, color='#2E86AB', alpha=0.8)
        plt.semilogx(output_freqs, output_mag_db, label='Model Output', linewidth=1.5, color='#A23B72', alpha=0.8)
        
        # Formatting
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Magnitude (dB)', fontsize=12)
        plt.title('Frequency Response Comparison: Target vs Model Output', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=11)
        
        # Set frequency range
        plt.xlim([20, 22000])
        
        # Set reasonable dB range (auto-scale based on data)
        all_mags = np.concatenate([target_mag_db, output_mag_db])
        mag_min = np.percentile(all_mags, 1)
        mag_max = np.percentile(all_mags, 99)
        plt.ylim([mag_min - 5, mag_max + 5])
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        elif show_plot:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot frequency response comparison between target and model output audio files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python plot_freq_response.py
  
  # Specify custom paths
  python plot_freq_response.py --target Data/val/dls2-target.wav --output Results/dls2-RNN3/best_val_out.wav
  
  # Save plot to file
  python plot_freq_response.py --save freq_response.png
        """
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='Data/val/dls2-target.wav',
        help='Path to target audio file (default: Data/val/dls2-target.wav)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='Results/dls2-RNN3/best_val_out.wav',
        help='Path to model output audio file (default: Results/dls2-RNN3/best_val_out.wav)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save plot (default: show interactively)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting, only show analysis'
    )
    
    args = parser.parse_args()
    
    try:
        plot_frequency_response(args.target, args.output, args.save, show_plot=not args.no_plot)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

