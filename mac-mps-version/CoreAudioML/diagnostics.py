"""
Training Dashboard with Diagnostic History

Clean, Apple-like dashboard for monitoring training progress with real-time
diagnostic plots and history tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import wavfile
from scipy import signal
import json
import os
from typing import Optional, List, Dict, Tuple


class TrainingDashboard:
    """Real-time training dashboard with diagnostic plots and history tracking."""
    
    # Apple-like color palette
    COLOR_TARGET = '#007AFF'  # iOS blue
    COLOR_MODEL = '#AF52DE'   # iOS purple
    COLOR_GRID = '#E5E5E5'    # Light gray
    COLOR_BG = '#FFFFFF'      # White
    COLOR_TEXT = '#1D1D1F'    # Dark gray/black
    
    def __init__(self, results_dir: str, max_history: int = 5):
        """
        Initialize training dashboard.
        
        Args:
            results_dir: Directory to save diagnostics and history
            max_history: Maximum number of best validations to track
        """
        self.results_dir = results_dir
        self.max_history = max_history
        self.history_file = os.path.join(results_dir, 'diagnostics_history.json')
        self.history: List[Dict] = []
        self.is_initialized = False  # Track if figure has been shown
        
        # Load existing history
        self._load_history()
        
        # Don't create figure yet - wait for first update
        self.fig = None
        self.axes = None
        self.spectrogram_colorbar = None  # Store colorbar reference to prevent duplicates
        self.validation_epochs = []  # Track which epochs validation occurred
        
    def _init_figure(self):
        """Initialize the dashboard figure with clean Apple-like design (called on first update)."""
        # Close any existing figures
        plt.close('all')
        
        # Create figure with clean white background
        self.fig = plt.figure(figsize=(16, 10), facecolor=self.COLOR_BG)
        self.fig.suptitle('Training Dashboard', fontsize=18, fontweight='600', 
                         color=self.COLOR_TEXT, y=0.98)
        
        # Create grid layout: 3 rows x 2 columns
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.08, right=0.95, top=0.94, bottom=0.06)
        
        self.axes = {
            'fft': self.fig.add_subplot(gs[0, 0]),
            'welch': self.fig.add_subplot(gs[0, 1]),
            'waveform': self.fig.add_subplot(gs[1, 0]),
            'spectrogram': self.fig.add_subplot(gs[1, 1]),
            'loss': self.fig.add_subplot(gs[2, 0]),
            'difference': self.fig.add_subplot(gs[2, 1])
        }
        
        # Apply clean styling to all axes
        for ax in self.axes.values():
            ax.set_facecolor(self.COLOR_BG)
            ax.grid(True, color=self.COLOR_GRID, linestyle='-', linewidth=0.5, alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.COLOR_GRID)
            ax.spines['bottom'].set_color(self.COLOR_GRID)
        
        # Don't show the figure yet - wait for first update
        # plt.show(block=False)
        # plt.pause(0.1)
    
    def _load_history(self):
        """Load diagnostic history from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                # Keep only last max_history entries
                self.history = self.history[-self.max_history:]
            except Exception as e:
                print(f"Warning: Could not load history: {e}")
                self.history = []
    
    def _save_history(self):
        """Save diagnostic history to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def _load_audio(self, filepath: str) -> Tuple[int, np.ndarray]:
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
            audio_data = np.mean(audio_data, axis=1)
        
        return sample_rate, audio_data
    
    def update(self, target_path: str, output_path: str, epoch: int, 
               val_loss: float, train_track: Dict):
        """
        Update dashboard with new best validation results.
        
        Args:
            target_path: Path to target audio file
            output_path: Path to model output audio file
            epoch: Current epoch number
            val_loss: Validation loss value
            train_track: Training tracking dictionary
        """
        # Load audio files
        target_sr, target_audio = self._load_audio(target_path)
        output_sr, output_audio = self._load_audio(output_path)
        
        # Ensure same sample rate and length
        if target_sr != output_sr:
            min_length = min(len(target_audio), len(output_audio))
            target_audio = target_audio[:min_length]
            output_audio = output_audio[:min_length]
            sample_rate = target_sr
        else:
            sample_rate = target_sr
            min_length = min(len(target_audio), len(output_audio))
            target_audio = target_audio[:min_length]
            output_audio = output_audio[:min_length]
        
        # Track validation epoch
        if epoch not in self.validation_epochs:
            self.validation_epochs.append(epoch)
            self.validation_epochs.sort()  # Keep sorted
        
        # Store in history
        history_entry = {
            'epoch': epoch,
            'val_loss': val_loss,
            'target_path': target_path,
            'output_path': output_path,
            'sample_rate': sample_rate
        }
        self.history.append(history_entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self._save_history()
        
        # Initialize figure on first update
        if not self.is_initialized:
            self._init_figure()
            self.is_initialized = True
        
        # Update all plots
        self._plot_all(target_audio, output_audio, sample_rate, epoch, val_loss, train_track)
        
        # Show/display the figure
        plt.show(block=False)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
    
    def _plot_all(self, target_audio: np.ndarray, output_audio: np.ndarray,
                  sample_rate: int, epoch: int, val_loss: float, train_track: Dict):
        """Update all subplots."""
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
            ax.set_facecolor(self.COLOR_BG)
            ax.grid(True, color=self.COLOR_GRID, linestyle='-', linewidth=0.5, alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.COLOR_GRID)
            ax.spines['bottom'].set_color(self.COLOR_GRID)
        
        # Plot each diagnostic
        self._plot_fft_comparison(target_audio, output_audio, sample_rate, epoch)
        self._plot_welch_comparison(target_audio, output_audio, sample_rate, epoch)
        self._plot_waveform_comparison(target_audio, output_audio, sample_rate)
        self._plot_spectrogram_comparison(target_audio, output_audio, sample_rate)
        self._plot_loss_history(train_track, epoch, val_loss)
        self._plot_difference(target_audio, output_audio, sample_rate, epoch)
    
    def _plot_fft_comparison(self, target_audio: np.ndarray, output_audio: np.ndarray,
                            sample_rate: int, epoch: int):
        """Plot classic FFT frequency response comparison."""
        ax = self.axes['fft']
        
        # Compute FFT
        target_fft = np.fft.rfft(target_audio)
        output_fft = np.fft.rfft(output_audio)
        
        # Get frequencies
        freqs = np.fft.rfftfreq(len(target_audio), 1/sample_rate)
        
        # Convert to magnitude in dB
        target_mag_db = 20 * np.log10(np.abs(target_fft) + 1e-12)
        output_mag_db = 20 * np.log10(np.abs(output_fft) + 1e-12)
        
        # Plot current
        ax.semilogx(freqs, target_mag_db, label='Target', color=self.COLOR_TARGET, 
                   linewidth=1.5, alpha=0.9)
        ax.semilogx(freqs, output_mag_db, label=f'Model (Epoch {epoch})', 
                   color=self.COLOR_MODEL, linewidth=1.5, alpha=0.9)
        
        # Plot history (with decreasing alpha)
        for i, entry in enumerate(self.history[:-1]):  # Exclude current
            try:
                _, hist_target = self._load_audio(entry['target_path'])
                _, hist_output = self._load_audio(entry['output_path'])
                min_len = min(len(hist_target), len(hist_output))
                hist_target = hist_target[:min_len]
                hist_output = hist_output[:min_len]
                
                hist_fft = np.fft.rfft(hist_output)
                hist_freqs = np.fft.rfftfreq(len(hist_output), 1/entry['sample_rate'])
                hist_mag_db = 20 * np.log10(np.abs(hist_fft) + 1e-12)
                
                alpha = 0.3 + 0.2 * (i / max(len(self.history) - 1, 1))
                ax.semilogx(hist_freqs, hist_mag_db, 
                           label=f'Epoch {entry["epoch"]}', 
                           color=self.COLOR_MODEL, linewidth=1, alpha=alpha, linestyle='--')
            except:
                pass
        
        ax.set_xlim([20, 22000])
        ax.set_xlabel('Frequency (Hz)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Magnitude (dB)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('FFT Frequency Response', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    def _plot_welch_comparison(self, target_audio: np.ndarray, output_audio: np.ndarray,
                              sample_rate: int, epoch: int):
        """Plot Welch's method frequency response comparison."""
        ax = self.axes['welch']
        
        # Compute Welch PSD
        target_freqs, target_psd = signal.welch(target_audio, fs=sample_rate, nperseg=2048,
                                               window='hann', noverlap=1024, average='mean')
        output_freqs, output_psd = signal.welch(output_audio, fs=sample_rate, nperseg=2048,
                                                window='hann', noverlap=1024, average='mean')
        
        # Convert to dB
        target_mag_db = 10 * np.log10(target_psd + 1e-12)
        output_mag_db = 10 * np.log10(output_psd + 1e-12)
        
        # Plot current
        ax.semilogx(target_freqs, target_mag_db, label='Target', color=self.COLOR_TARGET,
                   linewidth=1.5, alpha=0.9)
        ax.semilogx(output_freqs, output_mag_db, label=f'Model (Epoch {epoch})',
                   color=self.COLOR_MODEL, linewidth=1.5, alpha=0.9)
        
        # Plot history
        for i, entry in enumerate(self.history[:-1]):
            try:
                _, hist_output = self._load_audio(entry['output_path'])
                hist_freqs, hist_psd = signal.welch(hist_output, fs=entry['sample_rate'],
                                                   nperseg=2048, window='hann', noverlap=1024)
                hist_mag_db = 10 * np.log10(hist_psd + 1e-12)
                
                alpha = 0.3 + 0.2 * (i / max(len(self.history) - 1, 1))
                ax.semilogx(hist_freqs, hist_mag_db, 
                           label=f'Epoch {entry["epoch"]}',
                           color=self.COLOR_MODEL, linewidth=1, alpha=alpha, linestyle='--')
            except:
                pass
        
        ax.set_xlim([20, 22000])
        ax.set_xlabel('Frequency (Hz)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Magnitude (dB)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('Welch Frequency Response (Averaged)', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    def _plot_waveform_comparison(self, target_audio: np.ndarray, output_audio: np.ndarray,
                                 sample_rate: int):
        """Plot time-domain waveform comparison."""
        ax = self.axes['waveform']
        
        # Show first 2 seconds
        duration = 2.0
        samples = int(duration * sample_rate)
        samples = min(samples, len(target_audio), len(output_audio))
        
        time_axis = np.arange(samples) / sample_rate
        
        ax.plot(time_axis, target_audio[:samples], label='Target', 
               color=self.COLOR_TARGET, linewidth=1.5, alpha=0.8)
        ax.plot(time_axis, output_audio[:samples], label='Model', 
               color=self.COLOR_MODEL, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Amplitude', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('Waveform Comparison (First 2s)', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_xlim([0, duration])
    
    def _plot_spectrogram_comparison(self, target_audio: np.ndarray, output_audio: np.ndarray,
                                    sample_rate: int):
        """Plot spectrogram comparison."""
        ax = self.axes['spectrogram']
        
        # Use first 5 seconds for spectrogram
        duration = 5.0
        samples = int(duration * sample_rate)
        samples = min(samples, len(target_audio), len(output_audio))
        
        target_short = target_audio[:samples]
        output_short = output_audio[:samples]
        
        # Compute spectrograms
        target_freqs, target_times, target_spec = signal.spectrogram(
            target_short, fs=sample_rate, nperseg=2048, noverlap=1024, window='hann')
        output_freqs, output_times, output_spec = signal.spectrogram(
            output_short, fs=sample_rate, nperseg=2048, noverlap=1024, window='hann')
        
        # Convert to dB
        target_spec_db = 10 * np.log10(target_spec + 1e-12)
        output_spec_db = 10 * np.log10(output_spec + 1e-12)
        
        # Plot difference spectrogram (Model - Target)
        diff_spec = output_spec_db - target_spec_db
        
        im = ax.pcolormesh(target_times, target_freqs, diff_spec, 
                          shading='gouraud', cmap='RdYlBu_r', vmin=-20, vmax=20)
        
        ax.set_yscale('log')
        ax.set_ylim([20, 22000])
        # Set consistent x-axis limits to prevent shrinking
        ax.set_xlim([0, duration])
        ax.set_xlabel('Time (s)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Frequency (Hz)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('Spectrogram Difference (Model - Target)', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        
        # Remove existing colorbar if it exists to prevent duplicates
        if self.spectrogram_colorbar is not None:
            try:
                self.spectrogram_colorbar.remove()
            except:
                pass
            self.spectrogram_colorbar = None
        
        # Add colorbar (only one will be created now)
        self.spectrogram_colorbar = plt.colorbar(im, ax=ax)
        self.spectrogram_colorbar.set_label('Difference (dB)', fontsize=9, color=self.COLOR_TEXT)
    
    def _plot_loss_history(self, train_track: Dict, epoch: int, val_loss: float):
        """Plot training and validation loss history."""
        ax = self.axes['loss']
        
        # Determine max epoch for consistent x-axis
        max_epoch = max(
            train_track.get('current_epoch', 0),
            len(train_track.get('training_losses', [])),
            epoch
        )
        
        if 'training_losses' in train_track and len(train_track['training_losses']) > 0:
            train_losses = train_track['training_losses']
            epochs_train = range(1, len(train_losses) + 1)
            ax.plot(epochs_train, train_losses, label='Training Loss', 
                   color=self.COLOR_TARGET, linewidth=1.5, alpha=0.7)
        
        if 'validation_losses' in train_track and len(train_track['validation_losses']) > 0:
            val_losses = train_track['validation_losses']
            num_val_runs = len(val_losses)
            
            # Use tracked validation epochs if available, otherwise estimate
            if len(self.validation_epochs) >= num_val_runs:
                # Use the last N validation epochs (most recent)
                epochs_val = self.validation_epochs[-num_val_runs:]
            elif len(self.validation_epochs) > 0:
                # Partial tracking - use what we have and estimate the rest
                epochs_val = list(self.validation_epochs)
                # Estimate remaining epochs
                if len(epochs_val) < num_val_runs:
                    last_epoch = epochs_val[-1] if epochs_val else max_epoch
                    spacing = max(1, (max_epoch - last_epoch) / (num_val_runs - len(epochs_val)))
                    for i in range(len(epochs_val), num_val_runs):
                        epochs_val.append(int(last_epoch + (i - len(epochs_val) + 1) * spacing))
            else:
                # No tracking available - estimate proportionally
                if max_epoch > num_val_runs:
                    val_epoch_spacing = max_epoch / num_val_runs
                    epochs_val = [int((i + 1) * val_epoch_spacing) for i in range(num_val_runs)]
                else:
                    epochs_val = list(range(1, num_val_runs + 1))
            
            # Ensure epochs_val matches val_losses length
            if len(epochs_val) != num_val_runs:
                # Fallback: use proportional spacing
                if max_epoch > num_val_runs:
                    val_epoch_spacing = max_epoch / num_val_runs
                    epochs_val = [int((i + 1) * val_epoch_spacing) for i in range(num_val_runs)]
                else:
                    epochs_val = list(range(1, num_val_runs + 1))
            
            ax.plot(epochs_val, val_losses, label='Validation Loss', 
                   color=self.COLOR_MODEL, linewidth=1.5, alpha=0.7)
            
            # Mark best validation points
            if 'best_val_loss' in train_track:
                best_epochs = []
                best_losses = []
                current_best = float('inf')
                for i, vloss in enumerate(val_losses):
                    if vloss < current_best:
                        current_best = vloss
                        best_epochs.append(epochs_val[i])
                        best_losses.append(vloss)
                
                if best_epochs:
                    ax.scatter(best_epochs, best_losses, color='#FF3B30', s=50, 
                             zorder=5, label='Best Validation', marker='*')
        
        # Set consistent x-axis range
        if max_epoch > 0:
            ax.set_xlim([1, max(1, max_epoch)])
        
        ax.set_xlabel('Epoch', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Loss', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('Loss History', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_yscale('log')
    
    def _plot_difference(self, target_audio: np.ndarray, output_audio: np.ndarray,
                        sample_rate: int, epoch: int):
        """Plot frequency difference (Model - Target) in dB."""
        ax = self.axes['difference']
        
        # Compute Welch PSD
        target_freqs, target_psd = signal.welch(target_audio, fs=sample_rate, nperseg=2048,
                                               window='hann', noverlap=1024)
        output_freqs, output_psd = signal.welch(output_audio, fs=sample_rate, nperseg=2048,
                                                window='hann', noverlap=1024)
        
        # Convert to dB
        target_mag_db = 10 * np.log10(target_psd + 1e-12)
        output_mag_db = 10 * np.log10(output_psd + 1e-12)
        
        # Compute difference
        diff_db = output_mag_db - target_mag_db
        
        # Plot difference
        ax.semilogx(target_freqs, diff_db, color=self.COLOR_MODEL, linewidth=1.5, alpha=0.9)
        
        # Add zero line
        ax.axhline(y=0, color=self.COLOR_TEXT, linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add ±1dB and ±2dB thresholds
        ax.axhline(y=1, color=self.COLOR_GRID, linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(y=-1, color=self.COLOR_GRID, linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(y=2, color=self.COLOR_GRID, linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(y=-2, color=self.COLOR_GRID, linestyle=':', linewidth=0.8, alpha=0.5)
        
        # Fill excess/deficit regions
        has_excess = np.any(diff_db > 0)
        has_deficit = np.any(diff_db < 0)
        
        if has_excess:
            ax.fill_between(target_freqs, 0, diff_db, where=(diff_db > 0), 
                           color='#FF3B30', alpha=0.2, label='Excess')
        if has_deficit:
            ax.fill_between(target_freqs, 0, diff_db, where=(diff_db < 0), 
                           color='#34C759', alpha=0.2, label='Deficit')
        
        ax.set_xlim([20, 22000])
        ax.set_xlabel('Frequency (Hz)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_ylabel('Difference (dB)', fontsize=10, color=self.COLOR_TEXT)
        ax.set_title('Frequency Difference (Model - Target)', fontsize=12, fontweight='500', color=self.COLOR_TEXT)
        
        # Only show legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, color=self.COLOR_GRID, linestyle='-', linewidth=0.5, alpha=0.5)
    
    def close(self):
        """Close the dashboard figure."""
        if self.fig:
            plt.close(self.fig)

