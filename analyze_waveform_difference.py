"""
Analyze time-domain waveform differences between model output and target.
This helps diagnose if ESR loss is working correctly.
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import sys

def analyze_waveform_difference(model_path, target_path, output_dir=None):
    """
    Analyze time-domain waveform differences.
    
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
    
    # Calculate time-domain metrics
    mse = np.mean((target_audio - model_audio) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate ESR (Error-to-Signal Ratio)
    target_energy = np.mean(target_audio ** 2) + 1e-8
    esr = mse / target_energy
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(target_audio, model_audio)[0, 1]
    
    # Calculate normalized cross-correlation to find phase shift
    cross_corr = np.correlate(target_audio, model_audio, mode='full')
    max_corr_idx = np.argmax(cross_corr)
    phase_shift_samples = max_corr_idx - (len(model_audio) - 1)
    phase_shift_ms = (phase_shift_samples / target_sr) * 1000
    
    # Calculate amplitude difference
    target_rms = np.sqrt(np.mean(target_audio ** 2))
    model_rms = np.sqrt(np.mean(model_audio ** 2))
    amplitude_ratio = model_rms / (target_rms + 1e-8)
    amplitude_diff_db = 20 * np.log10(amplitude_ratio + 1e-8)
    
    # Calculate DC offset difference
    target_dc = np.mean(target_audio)
    model_dc = np.mean(model_audio)
    dc_diff = model_dc - target_dc
    
    print("=" * 70)
    print("WAVEFORM (TIME-DOMAIN) ANALYSIS")
    print("=" * 70)
    print(f"\nTarget file: {target_path}")
    print(f"Model file: {model_path}")
    print(f"Sample rate: {target_sr} Hz")
    print(f"Length: {min_len / target_sr:.2f} seconds ({min_len} samples)")
    
    print("\n--- Time-Domain Metrics ---")
    print(f"MSE (Mean Squared Error):     {mse:.6f}")
    print(f"RMSE (Root Mean Squared):     {rmse:.6f}")
    print(f"ESR (Error-to-Signal Ratio):  {esr:.6f}")
    print(f"Correlation Coefficient:      {correlation:.4f}")
    
    print("\n--- Amplitude Analysis ---")
    print(f"Target RMS:                   {target_rms:.6f}")
    print(f"Model RMS:                    {model_rms:.6f}")
    print(f"Amplitude Ratio:              {amplitude_ratio:.4f}")
    print(f"Amplitude Difference:       {amplitude_diff_db:.2f} dB")
    
    print("\n--- Phase Analysis ---")
    print(f"Phase Shift:                  {phase_shift_samples} samples ({phase_shift_ms:.2f} ms)")
    print(f"Max Cross-Correlation:        {np.max(cross_corr):.6f}")
    
    print("\n--- DC Offset Analysis ---")
    print(f"Target DC Offset:             {target_dc:.6f}")
    print(f"Model DC Offset:              {model_dc:.6f}")
    print(f"DC Difference:                {dc_diff:.6f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if correlation > 0.95:
        print("✓ Excellent waveform matching (correlation > 0.95)")
    elif correlation > 0.85:
        print("~ Good waveform matching (correlation > 0.85)")
    elif correlation > 0.70:
        print("⚠ Fair waveform matching (correlation > 0.70)")
    else:
        print("✗ Poor waveform matching (correlation < 0.70)")
        print("  → Consider increasing ESR loss weight")
    
    if abs(amplitude_diff_db) < 1.0:
        print("✓ Amplitude levels match well")
    elif abs(amplitude_diff_db) < 3.0:
        print("~ Amplitude levels are close")
    else:
        print(f"⚠ Significant amplitude difference ({amplitude_diff_db:.2f} dB)")
    
    if abs(phase_shift_ms) < 1.0:
        print("✓ No significant phase shift")
    else:
        print(f"⚠ Phase shift detected: {phase_shift_ms:.2f} ms")
    
    if abs(dc_diff) < 0.001:
        print("✓ DC offset matches well")
    else:
        print(f"⚠ DC offset difference: {dc_diff:.6f}")
        print("  → DC loss weight (0.10) may need adjustment")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Waveform comparison (first 0.1 seconds for detail)
    duration_detail = 0.1
    samples_detail = int(duration_detail * target_sr)
    samples_detail = min(samples_detail, min_len)
    time_detail = np.arange(samples_detail) / target_sr
    
    ax1 = axes[0]
    ax1.plot(time_detail, target_audio[:samples_detail], 
            label='Target', color='#007AFF', linewidth=2, alpha=0.8)
    ax1.plot(time_detail, model_audio[:samples_detail], 
            label='Model', color='#AF52DE', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Waveform Comparison (First {duration_detail*1000:.0f}ms) - Correlation: {correlation:.4f}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Waveform comparison (first 2 seconds for overview)
    duration_overview = 2.0
    samples_overview = int(duration_overview * target_sr)
    samples_overview = min(samples_overview, min_len)
    time_overview = np.arange(samples_overview) / target_sr
    
    ax2 = axes[1]
    ax2.plot(time_overview, target_audio[:samples_overview], 
            label='Target', color='#007AFF', linewidth=1.5, alpha=0.7)
    ax2.plot(time_overview, model_audio[:samples_overview], 
            label='Model', color='#AF52DE', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title(f'Waveform Comparison (First {duration_overview}s)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error signal
    error_signal = target_audio[:samples_overview] - model_audio[:samples_overview]
    ax3 = axes[2]
    ax3.plot(time_overview, error_signal, color='#FF3B30', linewidth=1.5, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.fill_between(time_overview, 0, error_signal, where=(error_signal > 0), 
                    color='#FF3B30', alpha=0.2)
    ax3.fill_between(time_overview, 0, error_signal, where=(error_signal < 0), 
                    color='#34C759', alpha=0.2)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Error (Target - Model)', fontsize=11)
    ax3.set_title(f'Error Signal (RMSE: {rmse:.6f}, ESR: {esr:.6f})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'waveform_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'esr': esr,
        'correlation': correlation,
        'amplitude_ratio': amplitude_ratio,
        'amplitude_diff_db': amplitude_diff_db,
        'phase_shift_samples': phase_shift_samples,
        'phase_shift_ms': phase_shift_ms,
        'dc_diff': dc_diff
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
    
    analyze_waveform_difference(model_path, target_path, output_dir=results_dir)

