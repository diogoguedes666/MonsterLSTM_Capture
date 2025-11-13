#!/usr/bin/env python3
"""
Comprehensive Training File Generator for LSTM Pedal Capture

This script generates at least 20 types of spectral sounds optimized for
capturing the characteristics of guitar pedals using LSTM neural networks.

Generated signals include:
- Various frequency sweeps (log, linear, exponential)
- Amplitude sweeps and modulations
- Noise types (white, pink, brown, band-limited)
- Harmonic series and chords
- Guitar-specific techniques
- Advanced test signals for nonlinear behavior
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PedalTrainingGenerator:
    """Generator for comprehensive pedal training signals"""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 base_duration: float = 1.0,
                 output_dir: str = "pedal_training_data",
                 filename_prefix: str = "pedal"):
        """
        Initialize the generator
        
        Args:
            sample_rate: Audio sample rate (default 44100 Hz)
            base_duration: Base duration for signals in seconds
            output_dir: Output directory for generated files
            filename_prefix: Prefix for output filenames
        """
        self.sample_rate = sample_rate
        self.base_duration = base_duration
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        
        # Create directory structure
        self.dirs = {
            'train': os.path.join(output_dir, 'train'),
            'val': os.path.join(output_dir, 'val'),
            'test': os.path.join(output_dir, 'test')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Metadata storage
        self.metadata = {
            'sample_rate': sample_rate,
            'base_duration': base_duration,
            'generation_date': datetime.now().isoformat(),
            'signals': []
        }
        
        # Guitar note frequencies (E2 to E4)
        self.guitar_notes = {
            'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00,
            'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
            'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56,
            'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00,
            'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
            'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
            'E4': 329.63
        }
    
    def _trim_trailing_silence(self, signal: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Remove trailing silence from signal"""
        if len(signal) == 0:
            return signal
        
        # Find the last non-silent sample
        abs_signal = np.abs(signal)
        # Find where signal drops below threshold
        non_silent_indices = np.where(abs_signal > threshold)[0]
        
        if len(non_silent_indices) > 0:
            # Return signal up to last non-silent sample, plus small fade-out
            last_index = non_silent_indices[-1]
            # Add small fade-out to avoid clicks (0.01 seconds)
            fade_samples = int(0.01 * self.sample_rate)
            end_index = min(last_index + fade_samples, len(signal))
            
            trimmed = signal[:end_index]
            # Apply fade-out
            if len(trimmed) > fade_samples:
                fade_out = np.linspace(1.0, 0.0, fade_samples)
                trimmed[-fade_samples:] *= fade_out
            
            return trimmed
        else:
            # If signal is all silence, return empty array
            return np.array([])
    
    def _remove_leading_silence(self, signal: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Remove leading silence from signal"""
        if len(signal) == 0:
            return signal
        
        abs_signal = np.abs(signal)
        non_silent_indices = np.where(abs_signal > threshold)[0]
        
        if len(non_silent_indices) > 0:
            first_index = non_silent_indices[0]
            return signal[first_index:]
        else:
            return np.array([])
    
    def _normalize(self, signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize signal to [-1, 1] range"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val * target_peak
        return signal
    
    def _apply_envelope(self, signal: np.ndarray, attack: float = 0.01, 
                       decay: float = 0.1, sustain: float = 0.7, 
                       release: float = 0.2) -> np.ndarray:
        """Apply ADSR envelope to signal"""
        samples = len(signal)
        envelope = np.ones(samples)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        if decay_samples > 0:
            decay_start = attack_samples
            decay_end = min(decay_start + decay_samples, samples - release_samples)
            decay_len = decay_end - decay_start
            if decay_len > 0:
                envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_len)
        
        # Sustain
        sustain_start = attack_samples + decay_samples
        sustain_end = samples - release_samples
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain
        
        # Release
        if release_samples > 0:
            release_start = max(0, samples - release_samples)
            envelope[release_start:] = np.linspace(sustain, 0, samples - release_start)
        
        return signal * envelope
    
    # ==================== FREQUENCY SWEEPS ====================
    
    def generate_log_sweep(self, f_start: float = 20.0, f_end: float = 20000.0, 
                          duration: Optional[float] = None) -> np.ndarray:
        """Logarithmic frequency sweep with multiple segments"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Split into multiple shorter segments for more variation
        num_segments = max(3, int(duration * 2))  # More segments for shorter duration
        segment_samples = samples // num_segments
        output = np.zeros(samples)
        
        for i in range(num_segments):
            seg_start = i * segment_samples
            seg_end = min((i + 1) * segment_samples, samples)
            seg_len = seg_end - seg_start
            t_seg = np.linspace(0, duration / num_segments, seg_len)
            
            # Vary frequency range for each segment
            freq_range = f_end - f_start
            seg_f_start = f_start + (freq_range * i / num_segments) + random.uniform(-freq_range*0.1, freq_range*0.1)
            seg_f_end = f_start + (freq_range * (i + 1) / num_segments) + random.uniform(-freq_range*0.1, freq_range*0.1)
            seg_f_start = max(20, min(20000, seg_f_start))
            seg_f_end = max(20, min(20000, seg_f_end))
            
            output[seg_start:seg_end] = signal.chirp(t_seg, seg_f_start, duration / num_segments, seg_f_end, method='logarithmic')
        
        return output
    
    def generate_linear_sweep(self, f_start: float = 20.0, f_end: float = 20000.0,
                              duration: Optional[float] = None) -> np.ndarray:
        """Linear frequency sweep with multiple segments"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Split into multiple shorter segments
        num_segments = max(3, int(duration * 2))
        segment_samples = samples // num_segments
        output = np.zeros(samples)
        
        for i in range(num_segments):
            seg_start = i * segment_samples
            seg_end = min((i + 1) * segment_samples, samples)
            seg_len = seg_end - seg_start
            t_seg = np.linspace(0, duration / num_segments, seg_len)
            
            # Vary frequency range for each segment
            freq_range = f_end - f_start
            seg_f_start = f_start + (freq_range * i / num_segments) + random.uniform(-freq_range*0.1, freq_range*0.1)
            seg_f_end = f_start + (freq_range * (i + 1) / num_segments) + random.uniform(-freq_range*0.1, freq_range*0.1)
            seg_f_start = max(20, min(20000, seg_f_start))
            seg_f_end = max(20, min(20000, seg_f_end))
            
            output[seg_start:seg_end] = signal.chirp(t_seg, seg_f_start, duration / num_segments, seg_f_end, method='linear')
        
        return output
    
    def generate_exponential_sweep(self, f_start: float = 20.0, f_end: float = 20000.0,
                                   duration: Optional[float] = None) -> np.ndarray:
        """Exponential frequency sweep"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        # Exponential sweep using quadratic method
        return signal.chirp(t, f_start, duration, f_end, method='quadratic')
    
    def generate_multiband_sweeps(self, duration: Optional[float] = None) -> np.ndarray:
        """Multi-band frequency sweeps (sub-bass, bass, mid, high)"""
        duration = duration or self.base_duration
        chunk_duration = duration / 4
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        bands = [
            (20, 60),      # Sub-bass
            (60, 250),     # Bass
            (250, 4000),   # Mid
            (4000, 20000)  # High
        ]
        
        for i, (f_start, f_end) in enumerate(bands):
            t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
            chunk = signal.chirp(t_chunk, f_start, chunk_duration, f_end, method='logarithmic')
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = chunk
        
        return output
    
    # ==================== AMPLITUDE VARIATIONS ====================
    
    def generate_amplitude_sweep_linear(self, base_signal: Optional[np.ndarray] = None,
                                       amp_start: float = 0.1, amp_end: float = 1.0,
                                       duration: Optional[float] = None) -> np.ndarray:
        """Linear amplitude sweep with varying frequencies"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Use multiple frequencies instead of single tone
        num_segments = max(3, int(duration * 2))
        segment_samples = samples // num_segments
        output = np.zeros(samples)
        
        guitar_freqs = list(self.guitar_notes.values())[:12]
        
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = min((i + 1) * segment_samples, samples)
            seg_len = end_idx - start_idx
            t_seg = t[start_idx:end_idx]
            
            # Vary frequency per segment
            freq = random.choice(guitar_freqs) * random.uniform(0.9, 1.1)
            base_signal = np.sin(2 * np.pi * freq * t_seg)
            
            # Add harmonics
            for h in range(2, random.randint(3, 6)):
                base_signal += (1.0 / h) * np.sin(2 * np.pi * freq * h * t_seg)
            
            # Create amplitude envelope for this segment
            seg_amp_start = amp_start + (amp_end - amp_start) * i / num_segments
            seg_amp_end = amp_start + (amp_end - amp_start) * (i + 1) / num_segments
            amp_envelope = np.linspace(seg_amp_start, seg_amp_end, seg_len)
            
            output[start_idx:end_idx] = base_signal * amp_envelope
        
        return output
    
    def generate_amplitude_sweep_log(self, base_signal: Optional[np.ndarray] = None,
                                     amp_start: float = 0.01, amp_end: float = 1.0,
                                     duration: Optional[float] = None) -> np.ndarray:
        """Logarithmic amplitude sweep"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        if base_signal is None:
            t = np.linspace(0, duration, samples)
            base_signal = np.sin(2 * np.pi * 440 * t)
        
        # Create logarithmic amplitude envelope
        amp_envelope = np.logspace(np.log10(amp_start), np.log10(amp_end), samples)
        return base_signal * amp_envelope
    
    def generate_chirp_with_am(self, f_start: float = 20.0, f_end: float = 20000.0,
                               mod_freq: float = 5.0, mod_depth: float = 0.5,
                               duration: Optional[float] = None) -> np.ndarray:
        """Chirp with amplitude modulation"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Generate chirp
        chirp_signal = signal.chirp(t, f_start, duration, f_end, method='logarithmic')
        
        # Apply amplitude modulation
        am_envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        return chirp_signal * am_envelope
    
    # ==================== NOISE GENERATORS ====================
    
    def generate_white_noise_bursts(self, burst_duration: float = 0.1,
                                    silence_duration: float = 0.1,
                                    duration: Optional[float] = None) -> np.ndarray:
        """White noise bursts with varying durations and guitar-like filtering"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Vary burst and silence durations
        output = np.zeros(samples)
        idx = 0
        
        while idx < samples:
            # Vary burst duration
            burst_dur = random.uniform(burst_duration * 0.5, burst_duration * 1.5)
            silence_dur = random.uniform(silence_duration * 0.5, silence_duration * 1.5)
            burst_samples = int(burst_dur * self.sample_rate)
            silence_samples = int(silence_dur * self.sample_rate)
            
            if idx + burst_samples <= samples:
                # Generate noise with guitar-like frequency shaping
                noise = np.random.normal(0, 0.3, burst_samples)
                # Apply bandpass filter to simulate guitar pickup response
                sos = signal.butter(4, [80, 8000], 'bandpass', fs=self.sample_rate, output='sos')
                noise = signal.sosfilt(sos, noise)
                output[idx:idx+burst_samples] = noise
            idx += burst_samples + silence_samples
        
        return output
    
    def generate_pink_noise(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Generate white noise
        white = np.random.normal(0, 1, samples)
        
        # Apply pink noise filter (simplified)
        # Using a more accurate pink noise generation
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        pink = signal.lfilter(b, a, white)
        
        return pink / np.max(np.abs(pink)) * 0.5
    
    def generate_brown_noise(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate brown noise (1/f² noise)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Brown noise is integrated white noise
        white = np.random.normal(0, 1, samples)
        brown = np.cumsum(white)
        
        return brown / np.max(np.abs(brown)) * 0.5
    
    def generate_bandlimited_noise(self, low_freq: float, high_freq: float,
                                   duration: Optional[float] = None) -> np.ndarray:
        """Band-limited noise"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Generate white noise
        noise = np.random.normal(0, 1, samples)
        
        # Apply bandpass filter
        sos = signal.butter(4, [low_freq, high_freq], 'bandpass', 
                           fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, noise)
        
        return filtered / np.max(np.abs(filtered)) * 0.5
    
    def generate_noise_bursts_multiband(self, duration: Optional[float] = None) -> np.ndarray:
        """Multiple band-limited noise bursts"""
        duration = duration or self.base_duration
        chunk_duration = duration / 4
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        bands = [
            (20, 200),      # Sub-bass
            (200, 2000),    # Low-mid
            (2000, 8000),   # High-mid
            (8000, 20000)   # High
        ]
        
        for i, (low, high) in enumerate(bands):
            chunk = self.generate_bandlimited_noise(low, high, chunk_duration)
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = chunk
        
        return output
    
    # ==================== IMPULSE AND TRANSIENT SIGNALS ====================
    
    def generate_impulse_train(self, interval: float = 0.1, 
                              amplitude_variation: bool = True,
                              duration: Optional[float] = None) -> np.ndarray:
        """Impulse train with varying intervals and amplitudes"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        interval_samples = int(interval * self.sample_rate)
        
        output = np.zeros(samples)
        idx = 0
        
        while idx < samples:
            if amplitude_variation:
                amp = random.uniform(0.5, 1.0)
            else:
                amp = 1.0
            output[idx] = amp
            idx += interval_samples
        
        return output
    
    def generate_transient_response(self, attack_times: List[float] = [0.001, 0.01, 0.05, 0.1],
                                    duration: Optional[float] = None) -> np.ndarray:
        """Transient response test with various attack times"""
        duration = duration or self.base_duration
        chunk_duration = duration / len(attack_times)
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        for i, attack_time in enumerate(attack_times):
            t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
            freq = 440.0
            signal_chunk = np.sin(2 * np.pi * freq * t_chunk)
            
            # Apply envelope with specific attack time
            attack_samples = int(attack_time * self.sample_rate)
            envelope = np.ones(samples_per_chunk)
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            envelope[attack_samples:] = np.exp(-np.linspace(0, 5, samples_per_chunk - attack_samples))
            
            signal_chunk *= envelope
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = signal_chunk
        
        return output
    
    # ==================== HARMONIC SERIES ====================
    
    def generate_harmonic_series(self, fundamental: float = 82.41, 
                                 num_harmonics: int = 15,
                                 duration: Optional[float] = None) -> np.ndarray:
        """Pure harmonic series with varying fundamentals and guitar-like characteristics"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Split into multiple notes with different fundamentals
        num_notes = max(2, int(duration * 3))  # More notes for variation
        note_samples = samples // num_notes
        output = np.zeros(samples)
        
        # Guitar note frequencies
        guitar_freqs = list(self.guitar_notes.values())[:12]  # First octave
        
        for i in range(num_notes):
            start_idx = i * note_samples
            end_idx = min((i + 1) * note_samples, samples)
            note_len = end_idx - start_idx
            t_note = np.linspace(0, duration / num_notes, note_len)
            
            # Random fundamental from guitar range
            fund = random.choice(guitar_freqs) * random.uniform(0.95, 1.05)  # Slight detuning
            
            signal_note = np.zeros(note_len)
            # Vary number of harmonics per note
            num_h = random.randint(3, min(num_harmonics, 8))
            for h in range(1, num_h + 1):
                amplitude = 1.0 / (h ** random.uniform(1.2, 1.8))  # Varying harmonic decay
                signal_note += amplitude * np.sin(2 * np.pi * fund * h * t_note)
            
            # Apply guitar-like envelope
            signal_note = self._apply_envelope(signal_note, 
                                             attack=random.uniform(0.001, 0.01),
                                             decay=random.uniform(0.05, 0.2),
                                             sustain=random.uniform(0.5, 0.8),
                                             release=random.uniform(0.05, 0.15))
            output[start_idx:end_idx] = signal_note
        
        return output
    
    def generate_inharmonic_series(self, fundamental: float = 82.41,
                                   stretch_factor: float = 1.02,
                                   num_harmonics: int = 15,
                                   duration: Optional[float] = None) -> np.ndarray:
        """Inharmonic series with stretched harmonics"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        signal_out = np.zeros(samples)
        for i in range(1, num_harmonics + 1):
            # Stretch harmonics
            freq = fundamental * i * (stretch_factor ** (i - 1))
            amplitude = 1.0 / i
            signal_out += amplitude * np.sin(2 * np.pi * freq * t)
        
        return signal_out
    
    def generate_power_chords(self, root_frequencies: List[float] = None,
                             duration: Optional[float] = None) -> np.ndarray:
        """Power chords (root + fifth) with more variation and guitar characteristics"""
        if root_frequencies is None:
            # Use more guitar frequencies with variation
            guitar_freqs = list(self.guitar_notes.values())[:12]
            root_frequencies = random.sample(guitar_freqs, min(4, len(guitar_freqs)))
        
        duration = duration or self.base_duration
        chunk_duration = duration / len(root_frequencies)
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        for i, root_freq in enumerate(root_frequencies):
            t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
            
            # Add slight detuning for realism
            root_freq_detuned = root_freq * random.uniform(0.98, 1.02)
            fifth_freq = root_freq_detuned * 1.5 * random.uniform(0.99, 1.01)
            
            # Root + fifth with varying ratios
            fifth_ratio = random.uniform(0.6, 0.8)
            chord = (np.sin(2 * np.pi * root_freq_detuned * t_chunk) + 
                    fifth_ratio * np.sin(2 * np.pi * fifth_freq * t_chunk))
            
            # Add varying harmonics
            num_harmonics = random.randint(2, 5)
            for harmonic in range(2, 2 + num_harmonics):
                amplitude = 1.0 / (harmonic ** random.uniform(1.3, 1.7))
                chord += amplitude * np.sin(2 * np.pi * root_freq_detuned * harmonic * t_chunk)
            
            # More realistic guitar envelope
            chord = self._apply_envelope(chord, 
                                       attack=random.uniform(0.005, 0.02),
                                       decay=random.uniform(0.05, 0.15),
                                       sustain=random.uniform(0.6, 0.85),
                                       release=random.uniform(0.1, 0.25))
            
            # Add slight noise for pick attack
            noise = np.random.normal(0, 0.01, len(chord))
            noise[:int(0.01 * self.sample_rate)] *= np.linspace(1, 0, int(0.01 * self.sample_rate))
            chord += noise
            
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = chord
        
        return output
    
    def generate_complex_chords(self, duration: Optional[float] = None) -> np.ndarray:
        """Complex chords (jazz/extended chords) with more variation"""
        duration = duration or self.base_duration
        
        # Use more chord variations
        num_chords = max(2, int(duration * 2))
        chunk_duration = duration / num_chords
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        # More diverse chord progressions
        all_chords = [
            [82.41, 103.83, 123.47, 164.81],  # E minor 7
            [110.00, 130.81, 164.81, 196.00],  # A minor 7
            [146.83, 174.61, 207.65, 246.94], # D major 7
            [196.00, 220.00, 261.63, 293.66],  # G major 9
            [82.41, 110.00, 146.83],          # Power chord progression
            [110.00, 138.59, 164.81],         # A minor
            [146.83, 174.61, 220.00],         # D major
            [196.00, 246.94, 293.66],        # G major
        ]
        
        # Select random chords
        selected_chords = random.sample(all_chords, min(num_chords, len(all_chords)))
        
        for i, chord_notes in enumerate(selected_chords):
            t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
            chord_signal = np.zeros_like(t_chunk)
            
            for j, freq in enumerate(chord_notes):
                # Add slight detuning
                freq_detuned = freq * random.uniform(0.99, 1.01)
                amplitude = 1.0 / (j + 1) * random.uniform(0.8, 1.2)
                chord_signal += amplitude * np.sin(2 * np.pi * freq_detuned * t_chunk)
                
                # Add harmonics
                num_h = random.randint(1, 3)
                for h in range(2, 2 + num_h):
                    chord_signal += (amplitude / h) * np.sin(2 * np.pi * freq_detuned * h * t_chunk)
            
            # Vary envelope per chord
            chord_signal = self._apply_envelope(chord_signal, 
                                              attack=random.uniform(0.01, 0.03),
                                              decay=random.uniform(0.1, 0.2),
                                              sustain=random.uniform(0.7, 0.9),
                                              release=random.uniform(0.2, 0.4))
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = chord_signal
        
        return output
    
    # ==================== GUITAR-SPECIFIC TECHNIQUES ====================
    
    def generate_vibrato(self, base_freq: float = 440.0, vibrato_rate: float = 5.0,
                        vibrato_depth: float = 0.02, duration: Optional[float] = None) -> np.ndarray:
        """Multiple notes with varying vibrato characteristics"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        
        # Split into multiple notes
        num_notes = max(2, int(duration * 2))
        note_samples = samples // num_notes
        output = np.zeros(samples)
        
        guitar_freqs = list(self.guitar_notes.values())[:12]
        
        for i in range(num_notes):
            start_idx = i * note_samples
            end_idx = min((i + 1) * note_samples, samples)
            note_len = end_idx - start_idx
            t_note = np.linspace(0, duration / num_notes, note_len)
            
            # Vary base frequency and vibrato parameters
            freq = random.choice(guitar_freqs) * random.uniform(0.95, 1.05)
            vibrato_rate_note = random.uniform(4.0, 7.0)
            vibrato_depth_note = random.uniform(0.01, 0.03)
            
            # Frequency modulation for vibrato
            freq_mod = vibrato_depth_note * np.sin(2 * np.pi * vibrato_rate_note * t_note)
            instantaneous_freq = freq * (1 + freq_mod)
            
            # Generate signal with frequency modulation
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
            signal_note = np.sin(phase)
            
            # Add harmonics
            for h in range(2, random.randint(3, 5)):
                phase_h = 2 * np.pi * np.cumsum(instantaneous_freq * h) / self.sample_rate
                signal_note += (1.0 / h) * np.sin(phase_h)
            
            # Apply envelope
            signal_note = self._apply_envelope(signal_note,
                                              attack=random.uniform(0.005, 0.015),
                                              release=random.uniform(0.05, 0.15))
            
            output[start_idx:end_idx] = signal_note
        
        return output
    
    def generate_palm_muted(self, base_freq: float = 82.41,
                           duration: Optional[float] = None) -> np.ndarray:
        """Palm-muted signal (low-pass filtered transients)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Generate signal with fast attack
        signal_out = np.sin(2 * np.pi * base_freq * t)
        
        # Apply fast decay envelope
        envelope = np.exp(-t * 10)
        signal_out *= envelope
        
        # Apply low-pass filter to simulate palm muting
        sos = signal.butter(4, 1000, 'lowpass', fs=self.sample_rate, output='sos')
        signal_out = signal.sosfilt(sos, signal_out)
        
        return signal_out
    
    def generate_pinch_harmonic(self, base_freq: float = 196.00,
                               duration: Optional[float] = None) -> np.ndarray:
        """Pinch harmonic with emphasized upper harmonics"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        signal_out = np.zeros(samples)
        
        # Emphasize higher harmonics (12th-20th)
        for harmonic in range(12, 21):
            amplitude = 1.0 / (harmonic - 10)  # Stronger than normal
            signal_out += amplitude * np.sin(2 * np.pi * base_freq * harmonic * t)
        
        # Add fundamental with lower amplitude
        signal_out += 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        return signal_out
    
    def generate_sliding_notes(self, start_freq: float = 82.41, end_freq: float = 110.00,
                               duration: Optional[float] = None) -> np.ndarray:
        """Sliding notes (frequency glide)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Linear frequency sweep
        return signal.chirp(t, start_freq, duration, end_freq, method='linear')
    
    def generate_tremolo_picking(self, base_freq: float = 146.83,
                                tremolo_rate: float = 16.0,
                                duration: Optional[float] = None) -> np.ndarray:
        """Tremolo picking (amplitude modulation)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Base signal
        signal_out = np.sin(2 * np.pi * base_freq * t)
        
        # Tremolo amplitude modulation
        tremolo = 0.5 * (1 + np.sin(2 * np.pi * tremolo_rate * t))
        signal_out *= tremolo
        
        return signal_out
    
    def generate_dynamic_playing(self, duration: Optional[float] = None) -> np.ndarray:
        """Dynamic playing with varying attack/release"""
        duration = duration or self.base_duration
        chunk_duration = duration / 5
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        techniques = [
            {'freq': 82.41, 'attack': 0.001, 'release': 0.05},   # Fast attack, quick release
            {'freq': 110.00, 'attack': 0.01, 'release': 0.1},    # Medium attack/release
            {'freq': 146.83, 'attack': 0.05, 'release': 0.2},   # Slow attack, longer release
            {'freq': 196.00, 'attack': 0.001, 'release': 0.3},  # Fast attack, long release
            {'freq': 246.94, 'attack': 0.02, 'release': 0.15}   # Medium attack/release
        ]
        
        for i, tech in enumerate(techniques):
            t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
            signal_chunk = np.sin(2 * np.pi * tech['freq'] * t_chunk)
            
            # Apply custom envelope
            envelope = self._apply_envelope(signal_chunk, 
                                          attack=tech['attack'],
                                          release=tech['release'])
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = envelope
        
        return output
    
    # ==================== ADVANCED TEST SIGNALS ====================
    
    def generate_phase_sweep(self, base_freq: float = 440.0,
                            duration: Optional[float] = None) -> np.ndarray:
        """Phase sweep (0 to 2π phase variations)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Sweep phase from 0 to 2π
        phase = np.linspace(0, 2 * np.pi, samples)
        signal_out = np.sin(2 * np.pi * base_freq * t + phase)
        
        return signal_out
    
    def generate_intermodulation_test(self, f1: float = 100.0, f2: float = 6000.0,
                                     duration: Optional[float] = None) -> np.ndarray:
        """Two-tone intermodulation distortion test"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Two-tone signal
        signal_out = (np.sin(2 * np.pi * f1 * t) + 
                     0.5 * np.sin(2 * np.pi * f2 * t))
        
        return signal_out
    
    def generate_crest_factor_variations(self, duration: Optional[float] = None) -> np.ndarray:
        """Crest factor variations (sine, square, impulse, noise)"""
        duration = duration or self.base_duration
        chunk_duration = duration / 4
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
        
        # Sine wave (low crest factor)
        output[:samples_per_chunk] = np.sin(2 * np.pi * 440 * t_chunk)
        
        # Square wave (medium crest factor)
        output[samples_per_chunk:2*samples_per_chunk] = signal.square(2 * np.pi * 440 * t_chunk) * 0.7
        
        # Impulse train (high crest factor)
        impulse_chunk = np.zeros(samples_per_chunk)
        impulse_chunk[::100] = 1.0
        output[2*samples_per_chunk:3*samples_per_chunk] = impulse_chunk
        
        # Random peaks (very high crest factor)
        noise_chunk = np.random.normal(0, 0.1, samples_per_chunk)
        peaks = np.random.randint(0, samples_per_chunk, 10)
        noise_chunk[peaks] = 1.0
        output[3*samples_per_chunk:] = noise_chunk
        
        return output
    
    def generate_nonlinear_test(self, duration: Optional[float] = None) -> np.ndarray:
        """Nonlinear test signals (amplitude-dependent content)"""
        duration = duration or self.base_duration
        chunk_duration = duration / 4
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate))
        
        t_chunk = np.linspace(0, chunk_duration, samples_per_chunk)
        
        # Different amplitude levels
        amplitudes = [0.1, 0.3, 0.6, 1.0]
        
        for i, amp in enumerate(amplitudes):
            # Generate signal with varying amplitude
            signal_chunk = amp * np.sin(2 * np.pi * 440 * t_chunk)
            
            # Add harmonics that become more prominent at higher amplitudes
            for harmonic in range(2, 5):
                harmonic_amp = amp * (1.0 / harmonic) * (amp ** 2)  # Nonlinear relationship
                signal_chunk += harmonic_amp * np.sin(2 * np.pi * 440 * harmonic * t_chunk)
            
            start_idx = i * samples_per_chunk
            end_idx = start_idx + samples_per_chunk
            output[start_idx:end_idx] = signal_chunk
        
        return output
    
    def generate_time_varying_spectra(self, duration: Optional[float] = None) -> np.ndarray:
        """Time-varying spectra (LFO-modulated signals)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Carrier frequency
        carrier_freq = 440.0
        
        # LFO frequencies for modulation
        lfo_freqs = [0.1, 0.5, 2.0, 5.0]
        chunk_samples = samples // len(lfo_freqs)
        output = np.zeros(samples)
        
        for i, lfo_freq in enumerate(lfo_freqs):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples if i < len(lfo_freqs) - 1 else samples
            t_chunk = t[start_idx:end_idx]
            
            # Frequency modulation
            mod = np.sin(2 * np.pi * lfo_freq * t_chunk)
            instantaneous_freq = carrier_freq * (1 + 0.1 * mod)
            
            # Generate FM signal
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
            output[start_idx:end_idx] = np.sin(phase)
        
        return output
    
    def generate_combined_signals(self, duration: Optional[float] = None) -> np.ndarray:
        """Combined signals (multiple techniques mixed)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Combine multiple signal types
        combined = np.zeros(samples)
        
        # Add chirp
        combined += 0.3 * self.generate_log_sweep(20, 20000, duration)
        
        # Add harmonic series
        combined += 0.3 * self.generate_harmonic_series(82.41, 10, duration)
        
        # Add noise bursts
        combined += 0.2 * self.generate_white_noise_bursts(0.1, 0.1, duration)
        
        # Add power chords
        combined += 0.2 * self.generate_power_chords(duration=duration)
        
        return combined
    
    def generate_realworld_guitar_patterns(self, duration: Optional[float] = None) -> np.ndarray:
        """Real-world guitar patterns (rhythmic patterns)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        output = np.zeros(samples)
        
        # Define rhythmic pattern (in beats)
        pattern = [
            (82.41, 0.25),   # E2, quarter note
            (110.00, 0.25),  # A2, quarter note
            (146.83, 0.5),   # D3, half note
            (82.41, 0.25),   # E2, quarter note
            (110.00, 0.25),  # A2, quarter note
            (146.83, 0.5),   # D3, half note
        ]
        
        bpm = 120
        beat_duration = 60.0 / bpm
        sample_idx = 0
        
        for freq, beats in pattern:
            note_duration = beats * beat_duration
            note_samples = int(note_duration * self.sample_rate)
            
            if sample_idx + note_samples <= samples:
                t_note = np.linspace(0, note_duration, note_samples)
                note_signal = np.sin(2 * np.pi * freq * t_note)
                
                # Apply envelope
                note_signal = self._apply_envelope(note_signal, attack=0.01, 
                                                  decay=0.05, sustain=0.8, 
                                                  release=0.1)
                output[sample_idx:sample_idx+note_samples] = note_signal
                sample_idx += note_samples
        
        return output
    
    # ==================== FILE OUTPUT ====================
    
    def save_signal(self, signal: np.ndarray, filename: str, 
                   split: str = 'train', normalize: bool = True) -> None:
        """
        Save signal to WAV file
        
        Args:
            signal: Audio signal array
            filename: Base filename (without extension)
            split: Dataset split ('train', 'val', 'test')
            normalize: Whether to normalize the signal
        """
        if normalize:
            signal = self._normalize(signal)
        
        filepath = os.path.join(self.dirs[split], f"{filename}.wav")
        wavfile.write(filepath, self.sample_rate, 
                     (signal * 32767).astype(np.int16))
    
    def generate_sub_bass_sweep(self, f_start: float = 20.0, f_end: float = 200.0,
                                duration: Optional[float] = None) -> np.ndarray:
        """Sub-bass frequency sweep (20Hz - 200Hz)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        return signal.chirp(t, f_start, duration, f_end, method='logarithmic')
    
    def generate_high_freq_sweep(self, f_start: float = 5000.0, f_end: float = 20000.0,
                                  duration: Optional[float] = None) -> np.ndarray:
        """High frequency sweep (5kHz - 20kHz)"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        return signal.chirp(t, f_start, duration, f_end, method='logarithmic')
    
    def generate_sub_bass_harmonics(self, fundamental: float = 40.0,
                                    num_harmonics: int = 10,
                                    duration: Optional[float] = None) -> np.ndarray:
        """Sub-bass harmonic series"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        signal_out = np.zeros(samples)
        for i in range(1, num_harmonics + 1):
            freq = fundamental * i
            if freq <= 200:  # Keep in sub-bass range
                amplitude = 1.0 / i
                signal_out += amplitude * np.sin(2 * np.pi * freq * t)
        
        return signal_out
    
    def generate_high_freq_harmonics(self, fundamental: float = 5000.0,
                                     num_harmonics: int = 4,
                                     duration: Optional[float] = None) -> np.ndarray:
        """High frequency harmonic series"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        signal_out = np.zeros(samples)
        for i in range(1, num_harmonics + 1):
            freq = fundamental * i
            if freq <= 20000:  # Keep in audible range
                amplitude = 1.0 / i
                signal_out += amplitude * np.sin(2 * np.pi * freq * t)
        
        return signal_out
    
    def generate_sub_high_combined(self, duration: Optional[float] = None) -> np.ndarray:
        """Combined sub-bass and high frequency content"""
        duration = duration or self.base_duration
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Split into alternating segments
        num_segments = max(4, int(duration * 2))
        segment_samples = samples // num_segments
        output = np.zeros(samples)
        
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = min((i + 1) * segment_samples, samples)
            seg_len = end_idx - start_idx
            t_seg = t[start_idx:end_idx]
            
            if i % 2 == 0:
                # Sub-bass segment
                freq = random.uniform(20, 200)
                segment = np.sin(2 * np.pi * freq * t_seg)
                # Add harmonics
                for h in range(2, random.randint(3, 6)):
                    if freq * h <= 200:
                        segment += (1.0 / h) * np.sin(2 * np.pi * freq * h * t_seg)
            else:
                # High frequency segment
                freq = random.uniform(5000, 20000)
                segment = np.sin(2 * np.pi * freq * t_seg)
                # Add harmonics
                for h in range(2, random.randint(2, 4)):
                    if freq * h <= 20000:
                        segment += (1.0 / h) * np.sin(2 * np.pi * freq * h * t_seg)
            
            output[start_idx:end_idx] = segment
        
        return output
    
    def generate_all_signals_sub_high(self, variations_per_type: int = 1,
                                     amplitude_levels: List[float] = None,
                                     max_total_duration: float = 120.0) -> Dict[str, List[np.ndarray]]:
        """
        Generate signals focused on sub-bass and high frequencies
        
        Args:
            variations_per_type: Number of variations per signal type
            amplitude_levels: List of amplitude levels to test
            max_total_duration: Maximum total duration in seconds
        
        Returns:
            Dictionary mapping signal type names to lists of signal arrays
        """
        if amplitude_levels is None:
            amplitude_levels = [0.3, 0.5, 0.7, 1.0]
        
        all_signals = {}
        max_total_samples = int(max_total_duration * self.sample_rate)
        
        # Define sub-bass and high frequency focused generators
        generators = [
            ('sub_bass_sweep', lambda d: self.generate_sub_bass_sweep(20, 200, d)),
            ('high_freq_sweep', lambda d: self.generate_high_freq_sweep(5000, 20000, d)),
            ('sub_bass_harmonics_40hz', lambda d: self.generate_sub_bass_harmonics(40, 10, d)),
            ('sub_bass_harmonics_60hz', lambda d: self.generate_sub_bass_harmonics(60, 8, d)),
            ('sub_bass_harmonics_80hz', lambda d: self.generate_sub_bass_harmonics(80, 6, d)),
            ('high_freq_harmonics_5k', lambda d: self.generate_high_freq_harmonics(5000, 4, d)),
            ('high_freq_harmonics_10k', lambda d: self.generate_high_freq_harmonics(10000, 2, d)),
            ('sub_high_combined', self.generate_sub_high_combined),
            ('sub_bass_noise', lambda d: self.generate_bandlimited_noise(20, 200, d)),
            ('high_freq_noise', lambda d: self.generate_bandlimited_noise(5000, 20000, d)),
            ('sub_bass_chirp', lambda d: signal.chirp(np.linspace(0, d, int(d*self.sample_rate)), 20, d, 200, method='logarithmic')),
            ('high_freq_chirp', lambda d: signal.chirp(np.linspace(0, d, int(d*self.sample_rate)), 5000, d, 20000, method='logarithmic')),
        ]
        
        # Generate signals
        current_total_samples = 0
        for signal_name, generator_func in generators:
            signals_list = []
            
            # Generate base signals first
            base_signals = []
            for variation in range(variations_per_type):
                try:
                    sig = generator_func(self.base_duration)
                    base_signals.append(sig)
                except Exception as e:
                    print(f"Warning: Error generating {signal_name} variation {variation}: {e}")
                    continue
            
            # Apply amplitude variations, but limit to fit in budget
            signals_added = 0
            for base_sig in base_signals:
                if current_total_samples >= max_total_samples:
                    break
                
                for amp_level in amplitude_levels:
                    if current_total_samples >= max_total_samples:
                        break
                    
                    sig_scaled = base_sig * amp_level
                    sig_samples = len(sig_scaled)
                    
                    # Check if we can add this signal
                    if current_total_samples + sig_samples <= max_total_samples:
                        signals_list.append(sig_scaled)
                        current_total_samples += sig_samples
                        signals_added += 1
                    else:
                        # Try to add partial if there's meaningful space
                        remaining_samples = max_total_samples - current_total_samples
                        if remaining_samples > self.sample_rate * 0.1:  # At least 0.1 seconds
                            sig_partial = sig_scaled[:remaining_samples]
                            signals_list.append(sig_partial)
                            current_total_samples += len(sig_partial)
                            signals_added += 1
                        break
                
                if current_total_samples >= max_total_samples:
                    break
            
            if signals_list:
                all_signals[signal_name] = signals_list
                self.metadata['signals'].append({
                    'name': signal_name,
                    'variations': len(signals_list),
                    'amplitude_levels': amplitude_levels,
                    'frequency_range': 'sub_bass_high'
                })
        
        total_duration = current_total_samples / self.sample_rate
        print(f"Generated {len(all_signals)} sub/high frequency signal types, total duration: {total_duration:.2f} seconds")
        
        return all_signals
    
    def generate_dataset_sub_high(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                                  test_ratio: float = 0.15, variations_per_type: int = 1,
                                  max_duration_seconds: float = 120.0,
                                  output_suffix: str = "_subhigh") -> None:
        """
        Generate dataset focused on sub-bass and high frequencies
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            variations_per_type: Number of variations per signal type
            max_duration_seconds: Maximum duration per split in seconds
            output_suffix: Suffix to add to output directory name
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Create separate output directory
        original_output_dir = self.output_dir
        self.output_dir = original_output_dir + output_suffix
        
        # Recreate directory structure
        self.dirs = {
            'train': os.path.join(self.output_dir, 'train'),
            'val': os.path.join(self.output_dir, 'val'),
            'test': os.path.join(self.output_dir, 'test')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print("Generating sub-bass and high frequency signal types...")
        all_signals = self.generate_all_signals_sub_high(
            variations_per_type=variations_per_type,
            max_total_duration=max_duration_seconds
        )
        
        max_samples_per_split = int(max_duration_seconds * self.sample_rate)
        num_signal_types = len(all_signals)
        
        # Get amplitude levels from metadata or use default
        amplitude_levels = [0.3, 0.5, 0.7, 1.0]  # Default
        if self.metadata.get('signals'):
            for sig_info in self.metadata['signals']:
                if 'amplitude_levels' in sig_info:
                    amplitude_levels = sig_info['amplitude_levels']
                    break
        
        print(f"Total signal types: {num_signal_types}")
        print(f"Max duration per split: {max_duration_seconds:.1f} seconds")
        
        # For each split, select signals ensuring all types are represented
        for split_name, split_ratio in [('train', train_ratio), 
                                        ('val', val_ratio), 
                                        ('test', test_ratio)]:
            combined_input = []
            combined_target = []
            current_samples = 0
            
            # Ensure we include at least one signal from each type
            for signal_name, signals in all_signals.items():
                if not signals:
                    continue
                
                # Select signals from this type, ensuring diversity
                available_signals = signals.copy()
                random.shuffle(available_signals)
                
                signals_added = 0
                for sig in available_signals:
                    sig_samples = len(sig)
                    
                    # Check if we can add this signal
                    if current_samples + sig_samples > max_samples_per_split:
                        # Try to add partial if there's meaningful space
                        remaining_samples = max_samples_per_split - current_samples
                        if remaining_samples > self.sample_rate * 0.1:  # At least 0.1 seconds
                            sig_partial = sig[:remaining_samples]
                            combined_input.append(sig_partial)
                            combined_target.append(sig_partial)
                            current_samples += remaining_samples
                        break
                    
                    combined_input.append(sig)
                    combined_target.append(sig)
                    current_samples += sig_samples
                    signals_added += 1
                    
                    # Limit to prevent one type from dominating
                    if signals_added >= len(amplitude_levels) and current_samples >= max_samples_per_split * 0.8:
                        break
                    
                    if current_samples >= max_samples_per_split:
                        break
                
                if current_samples >= max_samples_per_split:
                    break
            
            # Concatenate all signals
            if combined_input:
                final_input = np.concatenate(combined_input)
                final_target = np.concatenate(combined_target)
                
                # Remove leading silence
                final_input = self._remove_leading_silence(final_input)
                final_target = self._remove_leading_silence(final_target)
                
                # Trim trailing silence
                final_input = self._trim_trailing_silence(final_input)
                final_target = self._trim_trailing_silence(final_target)
                
                # Ensure we don't exceed max duration (safety check)
                if len(final_input) > max_samples_per_split:
                    final_input = final_input[:max_samples_per_split]
                    final_target = final_target[:max_samples_per_split]
                
                # Skip if signal is too short (all silence)
                if len(final_input) < self.sample_rate * 0.1:  # Less than 0.1 seconds
                    print(f"Warning: {split_name} set is too short after trimming, skipping...")
                    continue
                
                # Normalize
                final_input = self._normalize(final_input)
                final_target = self._normalize(final_target)
                
                # Save files
                input_filename = f"{self.filename_prefix}-input"
                target_filename = f"{self.filename_prefix}-target"
                
                self.save_signal(final_input, input_filename, split=split_name, normalize=False)
                self.save_signal(final_target, target_filename, split=split_name, normalize=False)
                
                duration_seconds = len(final_input) / self.sample_rate
                num_signals_used = len(combined_input)
                print(f"Saved {split_name} set: {num_signals_used} signals, "
                     f"{len(final_input)} samples ({duration_seconds:.2f} seconds, max: {max_duration_seconds:.0f}s)")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        self.metadata['max_duration_seconds'] = max_duration_seconds
        self.metadata['num_signal_types'] = num_signal_types
        self.metadata['frequency_focus'] = 'sub_bass_and_high_frequencies'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nSub/High frequency dataset generation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Restore original output directory
        self.output_dir = original_output_dir
        self.dirs = {
            'train': os.path.join(original_output_dir, 'train'),
            'val': os.path.join(original_output_dir, 'val'),
            'test': os.path.join(original_output_dir, 'test')
        }
    
    def generate_all_signals(self, variations_per_type: int = 1,
                           amplitude_levels: List[float] = None,
                           max_total_duration: float = 120.0) -> Dict[str, List[np.ndarray]]:
        """
        Generate all signal types with variations, ensuring total fits in max duration
        
        Args:
            variations_per_type: Number of variations per signal type
            amplitude_levels: List of amplitude levels to test
            max_total_duration: Maximum total duration in seconds (default: 120 = 2 minutes)
        
        Returns:
            Dictionary mapping signal type names to lists of signal arrays
        """
        if amplitude_levels is None:
            # Fewer amplitude levels to reduce file size
            amplitude_levels = [0.3, 0.5, 0.7, 1.0]  # Reduced to 4 levels
        
        all_signals = {}
        max_total_samples = int(max_total_duration * self.sample_rate)
        
        # Define all signal generators
        generators = [
            ('log_sweep', self.generate_log_sweep),
            ('linear_sweep', self.generate_linear_sweep),
            ('exponential_sweep', self.generate_exponential_sweep),
            ('multiband_sweeps', self.generate_multiband_sweeps),
            ('amplitude_sweep_linear', lambda d: self.generate_amplitude_sweep_linear(duration=d)),
            ('amplitude_sweep_log', lambda d: self.generate_amplitude_sweep_log(duration=d)),
            ('chirp_with_am', self.generate_chirp_with_am),
            ('white_noise_bursts', self.generate_white_noise_bursts),
            ('pink_noise', self.generate_pink_noise),
            ('brown_noise', self.generate_brown_noise),
            ('bandlimited_noise_sub', lambda d: self.generate_bandlimited_noise(20, 200, d)),
            ('bandlimited_noise_mid', lambda d: self.generate_bandlimited_noise(200, 2000, d)),
            ('bandlimited_noise_high', lambda d: self.generate_bandlimited_noise(2000, 8000, d)),
            ('noise_bursts_multiband', self.generate_noise_bursts_multiband),
            ('impulse_train', self.generate_impulse_train),
            ('harmonic_series', lambda d: self.generate_harmonic_series(82.41, 15, d)),
            ('inharmonic_series', lambda d: self.generate_inharmonic_series(82.41, 1.02, 15, d)),
            ('power_chords', lambda d: self.generate_power_chords(duration=d)),
            ('complex_chords', self.generate_complex_chords),
            ('vibrato', lambda d: self.generate_vibrato(440.0, 5.0, 0.02, d)),
            ('palm_muted', lambda d: self.generate_palm_muted(82.41, d)),
            ('pinch_harmonic', lambda d: self.generate_pinch_harmonic(196.00, d)),
            ('sliding_notes', lambda d: self.generate_sliding_notes(82.41, 110.00, d)),
            ('tremolo_picking', lambda d: self.generate_tremolo_picking(146.83, 16.0, d)),
            ('dynamic_playing', self.generate_dynamic_playing),
            ('phase_sweep', lambda d: self.generate_phase_sweep(440.0, d)),
            ('intermodulation_test', lambda d: self.generate_intermodulation_test(100.0, 6000.0, d)),
            ('crest_factor_variations', self.generate_crest_factor_variations),
            ('transient_response', lambda d: self.generate_transient_response(duration=d)),
            ('nonlinear_test', self.generate_nonlinear_test),
            ('time_varying_spectra', self.generate_time_varying_spectra),
            ('combined_signals', self.generate_combined_signals),
            ('realworld_guitar_patterns', self.generate_realworld_guitar_patterns),
        ]
        
        # Calculate how many signals we can fit per type
        num_types = len(generators)
        samples_per_signal = int(self.base_duration * self.sample_rate)
        # Reserve space for at least 1 signal per type, with all amplitude levels
        min_samples_per_type = samples_per_signal * len(amplitude_levels)
        max_samples_per_type = max_total_samples // num_types
        
        # Generate signals
        current_total_samples = 0
        for signal_name, generator_func in generators:
            signals_list = []
            
            # Generate base signals first
            base_signals = []
            for variation in range(variations_per_type):
                try:
                    sig = generator_func(self.base_duration)
                    base_signals.append(sig)
                except Exception as e:
                    print(f"Warning: Error generating {signal_name} variation {variation}: {e}")
                    continue
            
            # Apply amplitude variations, but limit to fit in budget
            signals_added = 0
            for base_sig in base_signals:
                if current_total_samples >= max_total_samples:
                    break
                
                for amp_level in amplitude_levels:
                    if current_total_samples >= max_total_samples:
                        break
                    
                    sig_scaled = base_sig * amp_level
                    sig_samples = len(sig_scaled)
                    
                    # Check if we can add this signal
                    if current_total_samples + sig_samples <= max_total_samples:
                        signals_list.append(sig_scaled)
                        current_total_samples += sig_samples
                        signals_added += 1
                    else:
                        # Try to add partial if there's meaningful space
                        remaining_samples = max_total_samples - current_total_samples
                        if remaining_samples > self.sample_rate * 0.1:  # At least 0.1 seconds
                            sig_partial = sig_scaled[:remaining_samples]
                            signals_list.append(sig_partial)
                            current_total_samples += len(sig_partial)
                            signals_added += 1
                        break
                
                if current_total_samples >= max_total_samples:
                    break
            
            if signals_list:
                all_signals[signal_name] = signals_list
                self.metadata['signals'].append({
                    'name': signal_name,
                    'variations': len(signals_list),
                    'amplitude_levels': amplitude_levels
                })
        
        total_duration = current_total_samples / self.sample_rate
        print(f"Generated {len(all_signals)} signal types, total duration: {total_duration:.2f} seconds")
        
        return all_signals
    
    def generate_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                       test_ratio: float = 0.15, variations_per_type: int = 1,
                       max_duration_seconds: float = 120.0) -> None:
        """
        Generate complete dataset with train/val/test splits
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            variations_per_type: Number of variations per signal type
            max_duration_seconds: Maximum duration per split in seconds (default: 120 = 2 minutes)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        print("Generating all signal types...")
        all_signals = self.generate_all_signals(
            variations_per_type=variations_per_type,
            max_total_duration=max_duration_seconds
        )
        
        max_samples_per_split = int(max_duration_seconds * self.sample_rate)
        num_signal_types = len(all_signals)
        
        # Get amplitude levels from metadata or use default
        amplitude_levels = [0.3, 0.5, 0.7, 1.0]  # Default
        if self.metadata.get('signals'):
            first_signal = self.metadata['signals'][0]
            if 'amplitude_levels' in first_signal:
                amplitude_levels = first_signal['amplitude_levels']
        
        print(f"Total signal types: {num_signal_types}")
        print(f"Max duration per split: {max_duration_seconds:.1f} seconds")
        
        # For each split, select signals ensuring all types are represented
        for split_name, split_ratio in [('train', train_ratio), 
                                        ('val', val_ratio), 
                                        ('test', test_ratio)]:
            combined_input = []
            combined_target = []
            current_samples = 0
            
            # Ensure we include at least one signal from each type
            for signal_name, signals in all_signals.items():
                if not signals:
                    continue
                
                # Select signals from this type, ensuring diversity
                # Shuffle to get random selection
                available_signals = signals.copy()
                random.shuffle(available_signals)
                
                signals_added = 0
                for sig in available_signals:
                    sig_samples = len(sig)
                    
                    # Check if we can add this signal
                    if current_samples + sig_samples > max_samples_per_split:
                        # Try to add partial if there's meaningful space
                        remaining_samples = max_samples_per_split - current_samples
                        if remaining_samples > self.sample_rate * 0.1:  # At least 0.1 seconds
                            sig_partial = sig[:remaining_samples]
                            combined_input.append(sig_partial)
                            combined_target.append(sig_partial)
                            current_samples += remaining_samples
                        break
                    
                    combined_input.append(sig)
                    combined_target.append(sig)
                    current_samples += sig_samples
                    signals_added += 1
                    
                    # Limit to prevent one type from dominating
                    if signals_added >= len(amplitude_levels) and current_samples >= max_samples_per_split * 0.8:
                        break
                    
                    if current_samples >= max_samples_per_split:
                        break
                
                if current_samples >= max_samples_per_split:
                    break
            
            # Concatenate all signals
            if combined_input:
                final_input = np.concatenate(combined_input)
                final_target = np.concatenate(combined_target)
                
                # Remove leading silence
                final_input = self._remove_leading_silence(final_input)
                final_target = self._remove_leading_silence(final_target)
                
                # Trim trailing silence
                final_input = self._trim_trailing_silence(final_input)
                final_target = self._trim_trailing_silence(final_target)
                
                # Ensure we don't exceed max duration (safety check)
                if len(final_input) > max_samples_per_split:
                    final_input = final_input[:max_samples_per_split]
                    final_target = final_target[:max_samples_per_split]
                
                # Skip if signal is too short (all silence)
                if len(final_input) < self.sample_rate * 0.1:  # Less than 0.1 seconds
                    print(f"Warning: {split_name} set is too short after trimming, skipping...")
                    continue
                
                # Normalize
                final_input = self._normalize(final_input)
                final_target = self._normalize(final_target)
                
                # Save files
                input_filename = f"{self.filename_prefix}-input"
                target_filename = f"{self.filename_prefix}-target"
                
                self.save_signal(final_input, input_filename, split=split_name, normalize=False)
                self.save_signal(final_target, target_filename, split=split_name, normalize=False)
                
                duration_seconds = len(final_input) / self.sample_rate
                num_signals_used = len(combined_input)
                print(f"Saved {split_name} set: {num_signals_used} signals, "
                     f"{len(final_input)} samples ({duration_seconds:.2f} seconds, max: {max_duration_seconds:.0f}s)")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        self.metadata['max_duration_seconds'] = max_duration_seconds
        self.metadata['num_signal_types'] = num_signal_types
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Main function to generate training files"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training files for LSTM pedal capture')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='Sample rate in Hz (default: 44100)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Base duration per signal in seconds (default: 1.0)')
    parser.add_argument('--output_dir', type=str, default='pedal_training_data',
                       help='Output directory (default: pedal_training_data)')
    parser.add_argument('--filename_prefix', type=str, default='pedal',
                       help='Filename prefix (default: pedal)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--max_duration', type=float, default=120.0,
                       help='Maximum duration per split in seconds (default: 120.0 = 2 minutes)')
    parser.add_argument('--variations', type=int, default=1,
                       help='Number of variations per signal type (default: 1)')
    parser.add_argument('--generate_subhigh', action='store_true',
                       help='Also generate sub-bass and high frequency focused dataset')
    
    args = parser.parse_args()
    
    # Create generator
    generator = PedalTrainingGenerator(
        sample_rate=args.sample_rate,
        base_duration=args.duration,
        output_dir=args.output_dir,
        filename_prefix=args.filename_prefix
    )
    
    # Generate main dataset
    generator.generate_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        variations_per_type=args.variations,
        max_duration_seconds=args.max_duration
    )
    
    # Generate sub/high frequency dataset if requested
    if args.generate_subhigh:
        print("\n" + "="*60)
        print("Generating sub-bass and high frequency focused dataset...")
        print("="*60)
        generator.generate_dataset_sub_high(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            variations_per_type=args.variations,
            max_duration_seconds=args.max_duration
        )


if __name__ == '__main__':
    main()

