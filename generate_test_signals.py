import numpy as np
from scipy import signal
import soundfile as sf
import os
from typing import List, Tuple, Dict
import random
from datetime import datetime
import json

class TestSignalGenerator:
    def __init__(self, 
                 sample_rate: int = 44100,
                 duration: float = 10.0,
                 base_dir: str = "dataset"):
        """
        Initialize the test signal generator
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Duration of each test signal in seconds
            base_dir: Base directory for all outputs
        """
        self.sr = sample_rate
        self.duration = duration
        self.base_dir = base_dir
        self.samples = int(self.sr * self.duration)
        self.time = np.linspace(0, duration, self.samples)
        
        # Create directory structure
        self.dirs = {
            'test_signals': os.path.join(base_dir, 'test_signals'),
            'dataset': {
                'train': os.path.join(base_dir, 'dataset', 'train'),
                'val': os.path.join(base_dir, 'dataset', 'val'),
                'test': os.path.join(base_dir, 'dataset', 'test')
            },
            'combined': os.path.join(base_dir, 'combined')
        }
        
        self._create_directories()
        
        # Store metadata
        self.metadata = {
            'sample_rate': sample_rate,
            'duration': duration,
            'generation_date': datetime.now().isoformat(),
            'signals': {}
        }

    def _create_directories(self):
        """Create all necessary directories"""
        for dir_path in [self.dirs['test_signals'], self.dirs['combined']] + list(self.dirs['dataset'].values()):
            os.makedirs(dir_path, exist_ok=True)

    def save_signal(self, signal: np.ndarray, filename: str) -> None:
        """Save the signal as a WAV file"""
        filepath = os.path.join(self.dirs['test_signals'], f"{filename}.wav")
        sf.write(filepath, signal, self.sr)

    def generate_sine_sweep(self, f_start: float = 20.0, f_end: float = 20000.0) -> np.ndarray:
        """Generate logarithmic sine sweep"""
        return signal.chirp(self.time, f_start, self.duration, f_end, method='logarithmic')

    def generate_guitar_chords(self) -> np.ndarray:
        """Generate synthetic guitar chord progressions with realistic characteristics"""
        # Standard guitar frequencies (E2 to E4)
        base_frequencies = {
            'E2': 82.41, 'A2': 110.00, 'D3': 146.83, 
            'G3': 196.00, 'B3': 246.94, 'E4': 329.63
        }
        
        signal = np.zeros(self.samples)
        chunk_size = self.samples // 4  # Split into 4 sections
        
        # Common chord progressions
        progressions = [
            [('E2', 'A2', 'D3'), ('A2', 'D3', 'G3'), ('G3', 'B3', 'E4'), ('E2', 'B3', 'E4')],  # Power chords
            [('E2', 'G3', 'B3'), ('A2', 'D3', 'G3'), ('D3', 'G3', 'B3'), ('E2', 'G3', 'B3')]   # Open chords
        ]
        
        for i, chord in enumerate(random.choice(progressions)):
            t = np.linspace(0, self.duration/4, chunk_size)
            chord_signal = np.zeros_like(t)
            
            # Generate each note in the chord
            for note in chord:
                f = base_frequencies[note]
                # Add fundamental
                chord_signal += np.sin(2 * np.pi * f * t)
                
                # Add harmonics with realistic decay
                for harmonic in range(2, 16):
                    amplitude = 1.0 / (harmonic ** 1.5)  # More realistic harmonic decay
                    chord_signal += amplitude * np.sin(2 * np.pi * f * harmonic * t)
            
            # Add slight pitch variations (vibrato)
            vibrato_rate = 5.0  # 5 Hz vibrato
            vibrato_depth = 0.003  # Subtle pitch variation
            vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            chord_signal *= vibrato
            
            # Add attack and decay envelope
            envelope = np.exp(-t * 2)  # Natural decay
            envelope[:int(0.01 * self.sr)] *= np.linspace(0, 1, int(0.01 * self.sr))  # Attack
            
            chord_signal *= envelope
            signal[i*chunk_size:(i+1)*chunk_size] = chord_signal
        
        return signal / np.max(np.abs(signal))

    def generate_impulses(self) -> np.ndarray:
        """Generate impulse train with varying amplitudes"""
        signal = np.zeros(self.samples)
        impulse_positions = np.linspace(0, self.samples, 50, dtype=int)[:-1]
        for pos in impulse_positions:
            signal[pos] = random.uniform(0.5, 1.0)
        return signal

    def generate_square_burst(self) -> np.ndarray:
        """Generate bursts of square waves at different frequencies"""
        output = np.zeros(self.samples)
        frequencies = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
        
        chunk_size = self.samples // len(frequencies)
        for i, freq in enumerate(frequencies):
            t = np.linspace(0, self.duration/len(frequencies), chunk_size)
            chunk = signal.square(2 * np.pi * freq * t)
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output * 0.7  # Reduce amplitude to prevent clipping

    def generate_noise_bursts(self) -> np.ndarray:
        """Generate filtered noise bursts"""
        noise_signal = np.zeros(self.samples)
        burst_length = int(0.1 * self.sr)  # 100ms bursts
        
        for i in range(0, self.samples, burst_length * 2):
            if i + burst_length <= self.samples:
                noise_signal[i:i+burst_length] = np.random.normal(0, 1, burst_length)
        
        # Apply bandpass filter
        sos = signal.butter(4, [80, 5000], 'bandpass', fs=self.sr, output='sos')
        return signal.sosfilt(sos, noise_signal)

    def generate_dynamic_ramp(self) -> np.ndarray:
        """Generate amplitude ramp with various frequencies"""
        t = np.linspace(0, 1, self.samples)
        ramp = t
        
        frequencies = [82.41, 110.00, 146.83, 196.00]
        signal = np.zeros_like(t)
        
        for f in frequencies:
            signal += np.sin(2 * np.pi * f * t)
        
        return signal * ramp

    def generate_crest_factor_test(self) -> np.ndarray:
        """Generate signals with varying crest factors"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 4
        
        # Generate different sections with varying crest factors
        for i in range(4):
            t = np.linspace(0, self.duration/4, chunk_size)
            if i == 0:
                # Sine wave (low crest factor)
                chunk = np.sin(2 * np.pi * 440 * t)
            elif i == 1:
                # Square wave (medium crest factor)
                chunk = signal.square(2 * np.pi * 440 * t)
            elif i == 2:
                # Impulse train (high crest factor)
                chunk = np.zeros_like(t)
                chunk[::100] = 1
            else:
                # Random peaks (very high crest factor)
                chunk = np.random.normal(0, 0.1, len(t))
                peaks = np.random.randint(0, len(t), 10)
                chunk[peaks] = 1
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
            
        return output / np.max(np.abs(output))

    def generate_harmonic_series(self) -> np.ndarray:
        """Generate pure harmonic series"""
        fundamental = 82.41  # E2
        signal = np.zeros(self.samples)
        
        for i in range(1, 16):  # 15 harmonics
            amplitude = 1.0 / i
            signal += amplitude * np.sin(2 * np.pi * fundamental * i * self.time)
            
        return signal / np.max(np.abs(signal))

    def generate_amplitude_modulation(self) -> np.ndarray:
        """Generate amplitude modulated signals"""
        carrier_freq = 220.0  # A3
        mod_freq = 10.0  # 10 Hz modulation
        
        carrier = np.sin(2 * np.pi * carrier_freq * self.time)
        modulator = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * self.time))
        
        return carrier * modulator

    def generate_phase_test(self) -> np.ndarray:
        """Generate phase-shifted test signals"""
        freq = 440.0
        signal = np.zeros(self.samples)
        
        # Generate signals with different phase shifts
        phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        chunk_size = self.samples // len(phases)
        
        for i, phase in enumerate(phases):
            t = np.linspace(0, self.duration/len(phases), chunk_size)
            chunk = np.sin(2 * np.pi * freq * t + phase)
            signal[i*chunk_size:(i+1)*chunk_size] = chunk
            
        return signal

    def generate_combined_file(self, signals: Dict[str, np.ndarray], filename: str) -> np.ndarray:
        """
        Combine multiple signals into one file with markers
        
        Args:
            signals: Dictionary of signal names and their arrays
            filename: Output filename
        """
        # Add marker tone between signals (1kHz, 0.5s)
        marker = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.5, int(0.5 * self.sr)))
        marker = marker * 0.3  # Reduce marker volume
        
        # Combine signals with markers
        combined = np.array([])
        signal_positions = {}
        current_position = 0
        
        for name, sig in signals.items():
            # Add marker
            if len(combined) > 0:
                combined = np.concatenate([combined, marker])
                current_position += len(marker)
            
            # Record position and duration
            signal_positions[name] = {
                'start_sample': current_position,
                'duration_samples': len(sig),
                'start_time': current_position / self.sr,
                'duration_seconds': len(sig) / self.sr
            }
            
            # Add signal
            combined = np.concatenate([combined, sig])
            current_position += len(sig)
        
        # Normalize
        combined = combined / np.max(np.abs(combined))
        
        # Save combined signal
        filepath = os.path.join(self.dirs['combined'], f"{filename}.wav")
        sf.write(filepath, combined, self.sr)
        
        # Save metadata
        metadata = {
            'filename': f"{filename}.wav",
            'sample_rate': self.sr,
            'total_duration': len(combined) / self.sr,
            'signals': signal_positions
        }
        
        with open(os.path.join(self.dirs['combined'], f"{filename}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return combined

    def generate_dataset(self, 
                        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                        variations_per_signal: int = 5) -> None:
        """
        Generate train/val/test datasets with variations
        
        Args:
            split_ratios: Ratios for train/val/test splits
            variations_per_signal: Number of variations to generate for each signal type
        """
        signals_collection = {}
        
        # Generate variations of each signal type
        for i in range(variations_per_signal):
            # Basic signals with slight variations
            signals = {
                f'sine_sweep_{i}': self.generate_sine_sweep(
                    f_start=20.0 * (1 + 0.1 * random.random()),
                    f_end=20000.0 * (1 + 0.1 * random.random())
                ),
                f'guitar_chords_{i}': self.generate_guitar_chords(),
                f'impulses_{i}': self.generate_impulses(),
                f'square_burst_{i}': self.generate_square_burst(),
                f'noise_bursts_{i}': self.generate_noise_bursts(),
                f'dynamic_ramp_{i}': self.generate_dynamic_ramp(),
                f'crest_factor_{i}': self.generate_crest_factor_test(),
                f'harmonic_series_{i}': self.generate_harmonic_series(),
                f'amplitude_mod_{i}': self.generate_amplitude_modulation(),
                f'phase_test_{i}': self.generate_phase_test()
            }
            signals_collection.update(signals)
        
        # Split signals into train/val/test
        all_keys = list(signals_collection.keys())
        random.shuffle(all_keys)
        
        n_total = len(all_keys)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        train_keys = all_keys[:n_train]
        val_keys = all_keys[n_train:n_train + n_val]
        test_keys = all_keys[n_train + n_val:]
        
        # Create combined files for each split
        splits = {
            'train': train_keys,
            'val': val_keys,
            'test': test_keys
        }
        
        for split_name, keys in splits.items():
            split_signals = {k: signals_collection[k] for k in keys}
            
            # Generate individual files
            for name, sig in split_signals.items():
                sf.write(
                    os.path.join(self.dirs['dataset'][split_name], f"{name}.wav"),
                    sig,
                    self.sr
                )
            
            # Generate combined file for this split
            self.generate_combined_file(
                split_signals,
                f"combined_{split_name}"
            )
        
        # Save dataset metadata
        dataset_metadata = {
            'split_ratios': {
                'train': split_ratios[0],
                'val': split_ratios[1],
                'test': split_ratios[2]
            },
            'variations_per_signal': variations_per_signal,
            'files': {
                'train': train_keys,
                'val': val_keys,
                'test': test_keys
            }
        }
        
        with open(os.path.join(self.base_dir, 'dataset_metadata.json'), 'w') as f:
            json.dump(dataset_metadata, f, indent=4)

    def generate_dynamic_playing(self) -> np.ndarray:
        """Generate guitar-like signals with varying dynamics and playing techniques"""
        output_signal = np.zeros(self.samples)
        chunk_size = self.samples // 5
        
        # Different playing techniques
        techniques = [
            'single_notes', 'palm_mute', 'pinch_harmonic', 
            'sliding', 'tremolo'
        ]
        
        for i, technique in enumerate(techniques):
            t = np.linspace(0, self.duration/5, chunk_size)
            chunk = np.zeros_like(t)
            
            if technique == 'single_notes':
                # Fast single note sequence
                notes = [82.41, 110.00, 146.83, 196.00]
                note_duration = len(t) // len(notes)
                for j, note in enumerate(notes):
                    idx = j * note_duration
                    chunk[idx:idx+note_duration] = np.sin(2 * np.pi * note * t[:note_duration])
                    
            elif technique == 'palm_mute':
                # Palm-muted power chord with heavy low-end
                f = 82.41  # E2
                chunk = np.sin(2 * np.pi * f * t)
                # Add muted characteristics
                chunk *= np.exp(-t * 10)  # Faster decay
                sos = signal.butter(4, 1000, 'lowpass', fs=self.sr, output='sos')
                chunk = signal.sosfilt(sos, chunk)
                
            elif technique == 'pinch_harmonic':
                # Pinch harmonic with prominent upper harmonics
                f = 196.00  # G3
                for harmonic in range(12, 20):  # Focus on higher harmonics
                    chunk += (1.0/harmonic) * np.sin(2 * np.pi * f * harmonic * t)
                    
            elif technique == 'sliding':
                # Sliding effect between notes
                f_start, f_end = 82.41, 110.00
                chunk = signal.chirp(t, f_start, self.duration/5, f_end, method='linear')
                
            elif technique == 'tremolo':
                # Tremolo picking
                f = 146.83  # D3
                tremolo_rate = 16  # 16Hz tremolo
                amplitude_mod = 0.5 * (1 + np.sin(2 * np.pi * tremolo_rate * t))
                chunk = np.sin(2 * np.pi * f * t) * amplitude_mod
            
            # Apply common guitar signal characteristics
            chunk = self._apply_guitar_characteristics(chunk)
            output_signal[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output_signal / np.max(np.abs(output_signal))

    def _apply_guitar_characteristics(self, signal: np.ndarray) -> np.ndarray:
        """Apply realistic guitar signal characteristics"""
        # Add slight noise
        noise = np.random.normal(0, 0.001, len(signal))
        signal += noise
        
        # Add subtle harmonics
        harmonics = np.zeros_like(signal)
        for h in range(2, 8):
            harmonics += (0.1 / h) * np.roll(signal, h)
        signal += harmonics
        
        # Add slight compression
        threshold = 0.7
        ratio = 4.0
        mask = np.abs(signal) > threshold
        signal[mask] = threshold + (np.abs(signal[mask]) - threshold) / ratio * np.sign(signal[mask])
        
        return signal

    def generate_all(self):
        """Generate all signals, combinations, and datasets"""
        signals = {
            'sine_sweep': self.generate_sine_sweep(),
            'guitar_chords': self.generate_guitar_chords(),
            'impulses': self.generate_impulses(),
            'square_burst': self.generate_square_burst(),
            'noise_bursts': self.generate_noise_bursts(),
            'dynamic_ramp': self.generate_dynamic_ramp(),
            'crest_factor': self.generate_crest_factor_test(),
            'harmonic_series': self.generate_harmonic_series(),
            'amplitude_mod': self.generate_amplitude_modulation(),
            'phase_test': self.generate_phase_test(),
            'dynamic_playing': self.generate_dynamic_playing(),
            'pickup_response': self.generate_pickup_response(),
            'dynamics_test': self.generate_dynamics_test(),
            'feedback_test': self.generate_feedback_test(),
            'reverb_test': self.generate_reverb_test(),
            'delay_test': self.generate_delay_test(),
            'modulation_test': self.generate_modulation_test(),
            'distortion_test': self.generate_distortion_test(),
            'stereo_test': self.generate_stereo_test(),
            'frequency_sweep_detailed': self.generate_frequency_sweep_detailed(),
            'transient_response_test': self.generate_transient_response_test(),
            'nonlinear_test': self.generate_nonlinear_test(),
            'time_varying_test': self.generate_time_varying_test(),
        }
        
        # Save individual signals
        for name, sig in signals.items():
            sf.write(
                os.path.join(self.dirs['test_signals'], f"{name}.wav"),
                sig,
                self.sr
            )
        
        # Generate combined file
        self.generate_combined_file(signals, "all_signals_combined")
        
        # Generate dataset
        self.generate_dataset()
        
        print("Generation complete! Directory structure:")
        print(f"""
        {self.base_dir}/
        ├── test_signals/      # Individual test signals
        ├── combined/          # Combined signals with metadata
        └── dataset/           # Training datasets
            ├── train/
            ├── val/
            └── test/
        """)

    def generate_pickup_response(self) -> np.ndarray:
        """Generate signals that emulate different pickup positions and types"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 3
        
        # Simulate different pickup positions (bridge, middle, neck)
        positions = [
            {'name': 'bridge', 'cutoff': 2000},
            {'name': 'middle', 'cutoff': 1000},
            {'name': 'neck', 'cutoff': 500}
        ]
        
        for i, pos in enumerate(positions):
            t = np.linspace(0, self.duration/3, chunk_size)
            # Generate rich harmonic content
            sig = np.zeros_like(t)
            fundamental = 82.41  # E2
            
            # Add harmonics with position-dependent weighting
            for h in range(1, 20):
                harmonic_amp = 1.0 / (h * (1 + i/2))  # Position-dependent decay
                sig += harmonic_amp * np.sin(2 * np.pi * fundamental * h * t)
            
            # Apply position-specific EQ using standard highpass filter
            sos = signal.butter(4, pos['cutoff'], 'highpass', fs=self.sr, output='sos')
            sig = signal.sosfilt(sos, sig)
            
            output[i*chunk_size:(i+1)*chunk_size] = sig
        
        return output / np.max(np.abs(output))

    def generate_dynamics_test(self) -> np.ndarray:
        """Generate signals to test dynamic response"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 4
        
        # Test different input levels and attack times
        levels = [0.1, 0.3, 0.6, 1.0]  # Different input levels
        
        for i, level in enumerate(levels):
            t = np.linspace(0, self.duration/4, chunk_size)
            # Generate complex tone
            sig = np.sin(2 * np.pi * 220 * t)  # Fundamental
            sig += 0.5 * np.sin(2 * np.pi * 440 * t)  # First harmonic
            sig += 0.25 * np.sin(2 * np.pi * 880 * t)  # Second harmonic
            
            # Apply envelope
            attack = np.linspace(0, 1, int(0.01 * self.sr))  # 10ms attack
            envelope = np.ones_like(t)
            envelope[:len(attack)] = attack
            
            sig = sig * envelope * level
            output[i*chunk_size:(i+1)*chunk_size] = sig
        
        return output

    def generate_feedback_test(self) -> np.ndarray:
        """Generate signals to test feedback behavior"""
        output = np.zeros(self.samples)
        t = self.time
        
        # Generate increasing feedback-prone frequencies
        fundamental = 196.0  # G3
        feedback_gain = np.linspace(0, 1, self.samples)
        
        # Main signal
        signal = np.sin(2 * np.pi * fundamental * t)
        
        # Add increasing feedback components
        feedback = 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)  # First harmonic feedback
        feedback += 0.3 * np.sin(2 * np.pi * fundamental * 3 * t)  # Second harmonic feedback
        
        output = signal + (feedback * feedback_gain)
        
        return output / np.max(np.abs(output))

    def generate_reverb_test(self) -> np.ndarray:
        """Generate signals to test reverb response"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 4
        
        # Test different types of impulses and sustaining sounds
        for i in range(4):
            t = np.linspace(0, self.duration/4, chunk_size)
            chunk = np.zeros_like(t)
            
            if i == 0:
                # Sharp impulse
                chunk[0] = 1.0
            elif i == 1:
                # Short burst
                burst_len = int(0.05 * self.sr)  # 50ms burst
                chunk[:burst_len] = np.sin(2 * np.pi * 440 * t[:burst_len])
                chunk[:burst_len] *= np.hanning(burst_len)
            elif i == 2:
                # Sustained note with sudden cutoff
                chunk = np.sin(2 * np.pi * 440 * t)
                chunk[chunk_size//2:] = 0
            else:
                # Complex chord with natural decay
                freqs = [220, 277.18, 329.63]  # A3, C#4, E4
                for f in freqs:
                    chunk += np.sin(2 * np.pi * f * t)
                chunk *= np.exp(-t * 2)
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

    def generate_delay_test(self) -> np.ndarray:
        """Generate signals to test delay effects"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 3
        
        for i in range(3):
            t = np.linspace(0, self.duration/3, chunk_size)
            chunk = np.zeros_like(t)
            
            if i == 0:
                # Single notes with space for delay
                notes = [440, 554.37, 659.25]  # A4, C#5, E5
                note_len = chunk_size // 6
                for j, f in enumerate(notes):
                    start = j * note_len * 2  # Leave space between notes
                    chunk[start:start+note_len] = np.sin(2 * np.pi * f * t[:note_len])
                    chunk[start:start+note_len] *= np.hanning(note_len)
            elif i == 1:
                # Rhythmic pattern
                pattern_len = int(0.2 * self.sr)  # 200ms pattern
                pattern = np.sin(2 * np.pi * 440 * t[:pattern_len])
                pattern *= np.hanning(pattern_len)
                for j in range(0, chunk_size-pattern_len, pattern_len*2):
                    chunk[j:j+pattern_len] = pattern
            else:
                # Continuous melody with varying frequencies
                freq = 440 * np.exp(np.sin(2 * np.pi * 0.5 * t))  # Frequency modulation
                chunk = np.sin(2 * np.pi * freq * t)
                chunk *= np.hanning(len(chunk))
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

    def generate_modulation_test(self) -> np.ndarray:
        """Generate signals to test modulation effects (chorus, phaser, flanger)"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 3
        
        for i in range(3):
            t = np.linspace(0, self.duration/3, chunk_size)
            chunk = np.zeros_like(t)
            
            if i == 0:
                # For chorus: sustained notes with slight detuning
                base_freq = 440
                detuned_freqs = [base_freq * x for x in [0.99, 1.0, 1.01]]
                for f in detuned_freqs:
                    chunk += np.sin(2 * np.pi * f * t)
            elif i == 1:
                # For phaser: sweeping harmonics
                fundamental = 220
                for harmonic in range(1, 6):
                    phase = np.sin(2 * np.pi * 0.5 * t)  # Sweeping phase
                    chunk += np.sin(2 * np.pi * fundamental * harmonic * t + phase)
            else:
                # For flanger: time-varying delayed signal
                base = np.sin(2 * np.pi * 440 * t)
                mod_depth = int(0.002 * self.sr)  # 2ms maximum delay
                for j in range(len(t)):
                    delay = int(mod_depth * (1 + np.sin(2 * np.pi * 0.5 * t[j])))
                    if j + delay < len(base):
                        chunk[j] = base[j] + 0.7 * base[j+delay]
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

    def generate_distortion_test(self) -> np.ndarray:
        """Generate signals to test various distortion characteristics"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 4
        
        for i in range(4):
            t = np.linspace(0, self.duration/4, chunk_size)
            chunk = np.zeros_like(t)
            
            if i == 0:
                # Clean to slightly driven
                amplitude = np.linspace(0.5, 2.0, len(t))
                chunk = np.sin(2 * np.pi * 220 * t) * amplitude
            elif i == 1:
                # Multi-frequency content
                freqs = [82.41, 110.00, 146.83]  # Power chord frequencies
                for f in freqs:
                    chunk += np.sin(2 * np.pi * f * t)
                chunk *= 1.5  # Push it into overdrive
            elif i == 2:
                # Dynamic playing
                base = np.sin(2 * np.pi * 110 * t)
                envelope = np.zeros_like(t)
                for j in range(4):
                    start = j * len(t)//4
                    env = np.exp(-np.linspace(0, 5, len(t)//4))
                    envelope[start:start+len(t)//4] = env
                chunk = base * envelope * 2
            else:
                # High gain characteristics
                chunk = np.sin(2 * np.pi * 82.41 * t)  # Fundamental
                for harmonic in range(2, 6):
                    chunk += (1.0/harmonic) * np.sin(2 * np.pi * 82.41 * harmonic * t)
                chunk *= 3.0  # Push well into saturation
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

    def generate_stereo_test(self) -> np.ndarray:
        """Generate signals to test stereo effects (panning, stereo width, mid-side processing)"""
        # Modified to return mono signal for compatibility
        output = np.zeros(self.samples)
        t = self.time
        
        # Create a mono-compatible test signal that simulates stereo movement
        # Using amplitude modulation to simulate panning
        lfo = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz LFO
        signal = np.sin(2 * np.pi * 440 * t)
        output = signal * lfo
        
        return output

    def generate_frequency_sweep_detailed(self) -> np.ndarray:
        """Generate more detailed frequency response tests"""
        output = np.zeros(self.samples)
        chunk_size = self.samples // 4
        
        for i in range(4):
            t = np.linspace(0, self.duration/4, chunk_size)
            if i == 0:
                # Sub-bass frequencies (20-60 Hz)
                output[i*chunk_size:(i+1)*chunk_size] = signal.chirp(
                    t, f0=20, f1=60, t1=self.duration/4, method='logarithmic')
            elif i == 1:
                # Bass frequencies (60-250 Hz)
                output[i*chunk_size:(i+1)*chunk_size] = signal.chirp(
                    t, f0=60, f1=250, t1=self.duration/4, method='logarithmic')
            elif i == 2:
                # Mid frequencies (250-4000 Hz)
                output[i*chunk_size:(i+1)*chunk_size] = signal.chirp(
                    t, f0=250, f1=4000, t1=self.duration/4, method='logarithmic')
            else:
                # High frequencies (4000-20000 Hz)
                output[i*chunk_size:(i+1)*chunk_size] = signal.chirp(
                    t, f0=4000, f1=20000, t1=self.duration/4, method='logarithmic')
        
        return output

    def generate_transient_response_test(self) -> np.ndarray:
        """Generate signals to test transient response and attack/release behavior"""
        output = np.zeros(self.samples)
        t = self.time
        
        # Different attack times
        attack_times = [0.001, 0.01, 0.05, 0.1]  # seconds
        decay_times = [0.05, 0.1, 0.2, 0.4]  # seconds
        chunk_size = self.samples // len(attack_times)
        
        for i, (attack, decay) in enumerate(zip(attack_times, decay_times)):
            t_chunk = np.linspace(0, self.duration/len(attack_times), chunk_size)
            env = np.zeros_like(t_chunk)
            
            attack_samples = int(attack * self.sr)
            decay_samples = int(decay * self.sr)
            
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
            env[attack_samples:attack_samples+decay_samples] = np.exp(-np.linspace(0, 5, decay_samples))
            
            signal = np.sin(2 * np.pi * 440 * t_chunk) * env
            output[i*chunk_size:(i+1)*chunk_size] = signal
        
        return output

    def generate_nonlinear_test(self) -> np.ndarray:
        """Generate signals to test nonlinear behaviors"""
        output = np.zeros(self.samples)
        t = self.time
        chunk_size = self.samples // 4
        
        for i in range(4):
            if i == 0:
                # Intermodulation distortion test
                f1, f2 = 100, 6000
                chunk = np.sin(2 * np.pi * f1 * t[:chunk_size]) + np.sin(2 * np.pi * f2 * t[:chunk_size])
            elif i == 1:
                # Amplitude-dependent nonlinearity
                freq = 440
                amp = np.linspace(0.1, 2.0, chunk_size)
                chunk = amp * np.sin(2 * np.pi * freq * t[:chunk_size])
            elif i == 2:
                # Multiple harmonically-related frequencies
                fundamental = 110
                chunk = sum(1/h * np.sin(2 * np.pi * fundamental * h * t[:chunk_size]) 
                           for h in range(1, 8))
            else:
                # Dynamic nonlinearity test
                freq = 220
                chunk = np.sin(2 * np.pi * freq * t[:chunk_size])
                chunk *= (1 + 0.5 * np.sin(2 * np.pi * 2 * t[:chunk_size]))
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

    def generate_time_varying_test(self) -> np.ndarray:
        """Generate signals to test time-varying effects"""
        output = np.zeros(self.samples)
        t = self.time
        
        # LFO frequencies for modulation
        lfo_freqs = [0.1, 0.5, 2.0, 5.0]
        chunk_size = self.samples // len(lfo_freqs)
        
        for i, lfo_freq in enumerate(lfo_freqs):
            # Carrier signal
            carrier_freq = 440
            # Modulation
            mod = np.sin(2 * np.pi * lfo_freq * t[:chunk_size])
            # Apply different modulation types
            if i == 0:
                # Amplitude modulation
                chunk = np.sin(2 * np.pi * carrier_freq * t[:chunk_size]) * (1 + mod)
            elif i == 1:
                # Frequency modulation
                chunk = np.sin(2 * np.pi * carrier_freq * t[:chunk_size] + 3 * mod)
            elif i == 2:
                # Phase modulation
                chunk = np.sin(2 * np.pi * carrier_freq * t[:chunk_size] + mod)
            else:
                # Complex modulation
                chunk = np.sin(2 * np.pi * carrier_freq * t[:chunk_size] * (1 + 0.1 * mod)) * (1 + 0.5 * mod)
            
            output[i*chunk_size:(i+1)*chunk_size] = chunk
        
        return output / np.max(np.abs(output))

if __name__ == "__main__":
    # Create generator instance
    generator = TestSignalGenerator(
        sample_rate=44100,
        duration=10.0,
        base_dir="guitar_amp_dataset"
    )
    
    # Generate everything
    generator.generate_all() 