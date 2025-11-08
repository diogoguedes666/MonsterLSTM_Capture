import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import os
import csv
from scipy.io.wavfile import write
import warnings
import sys
import numpy as np
import psutil
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, fields
import logging
import yaml
from collections import defaultdict
import librosa
import random
from functools import wraps
import torch.nn.functional as F
import collections
import scipy.signal
import torchaudio
import torch.nn as nn
import gc
from concurrent.futures import ThreadPoolExecutor
from torch.nn.parallel import DistributedDataParallel
import asyncio
import torch.distributed as dist
import torch.profiler

#
warnings.filterwarnings("ignore", category=UserWarning)

prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. This is
    intended to recreate the training of models of the ht1 amplifier and big muff distortion pedal, but can easily be 
    adapted to use any dataset''')

# arguments for the training/test data locations and file names and config loading
prsr.add_argument('--device', '-p', default='dls2', help='This label describes what device is being modelled')
prsr.add_argument('--data_location', '-dl', default='./Data', help='Location of the "Data" directory')
prsr.add_argument('--file_name', '-fn', default='dls2',
                  help='The filename of the wav file to be loaded as the input/target data, the script looks for files'
                       'with the filename and the extensions -input.wav and -target.wav ')
prsr.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults"
                  , default='RNN3')
prsr.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
prsr.add_argument('--save_location', '-sloc', default='Results', help='Directory where trained models will be saved')
prsr.add_argument('--load_model', '-lm', default=True, help='load a pretrained model if it is found')

# pre-processing of the training/val/test data
prsr.add_argument('--segment_length', '-slen', type=int, default=22050, help='Training audio segment length in samples')

# number of epochs and validation
prsr.add_argument('--epochs', '-eps', type=int, default=2000, help='Max number of training epochs to run')
prsr.add_argument('--validation_f', '-vfr', type=int, default=2, help='Validation Frequency (in epochs)')
# TO DO
prsr.add_argument('--validation_p', '-vp', type=int, default=20,
                  help='How many validations without improvement before stopping training, None for no early stopping')

# settings for the training epoch
prsr.add_argument('--batch_size', '-bs', type=int, default=1024, help='Training mini-batch size')
prsr.add_argument('--iter_num', '-it', type=int, default=None,
                  help='Overrides --batch_size and instead sets the batch_size so that a total of --iter_num batches'
                       'are processed in each epoch')
prsr.add_argument('--learn_rate', '-lr', type=float, default=0.005, help='Initial learning rate')
prsr.add_argument('--init_len', '-il', type=int, default=200,
                  help='Number of sequence samples to process before starting weight updates')
prsr.add_argument('--up_fr', '-uf', type=int, default=750,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')
prsr.add_argument('--cuda', '-cu', default=1, help='Use GPU if available')

# loss function/s
prsr.add_argument('--loss_fcns', '-lf', default={'ESRPre': 0.70, 'DC': 0.10, 'Spectral': 0.20},
                  help='Which loss functions, ESR, ESRPre, DC, Spectral. Argument is a dictionary with each key representing a'
                       'loss function name and the corresponding value being the multiplication factor applied to that'
                       'loss function, used to control the contribution of each loss function to the overall loss ')
prsr.add_argument('--pre_filt',   '-pf',   default='None',
                    help='FIR filter coefficients for pre-emphasis filter, can also read in a csv file')

# the validation and test sets are divided into shorter chunks before processing to reduce the amount of GPU memory used
# you can probably ignore this unless during training you get a 'cuda out of memory' error
prsr.add_argument('--val_chunk', '-vs', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')
prsr.add_argument('--test_chunk', '-tc', type=int, default=100000, help='Number of sequence samples to process'
                                                                               'in each chunk of validation ')

# arguments for the network structure
prsr.add_argument('--model', '-m', default='SimpleRNN', type=str, help='model architecture')
prsr.add_argument('--input_size', '-is', default=1, type=int, help='1 for mono input data, 2 for stereo, etc ')
prsr.add_argument('--output_size', '-os', default=1, type=int, help='1 for mono output data, 2 for stereo, etc ')
prsr.add_argument('--num_blocks', '-nb', default=2, type=int, help='Number of recurrent blocks')
prsr.add_argument('--hidden_size', '-hs', default=32, type=int, help='Recurrent unit hidden state size')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU or RNN')
prsr.add_argument('--skip_con', '-sc', default=1, help='is there a skip connection for the input to the output')

prsr.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='Weight decay for optimizer')
prsr.add_argument('--gradient_clip', '-gc', type=float, default=1.0, help='Gradient clipping value')

args = prsr.parse_args()

# This function takes a directory as argument, looks for an existing model file called 'model.json' and loads a network
# from it, after checking the network in 'model.json' matches the architecture described in args. If no model file is
# found, it creates a network according to the specification in args.
def init_model(save_path, args):
    if miscfuncs.file_check('model.json', save_path) and args.load_model:
        print('existing model file found, loading network')
        model_data = miscfuncs.json_load('model', save_path)
        network = networks.load_model(model_data)
    else:
        print('no saved model found, creating new network')
        network = networks.SimpleRNN(
            input_size=args.input_size, 
            unit_type=args.unit_type, 
            hidden_size=args.hidden_size,
            output_size=args.output_size, 
            skip=args.skip_con
        )
    return network

@dataclass
class AutoTunerConfig:
    """Enhanced configuration class for AutoTuner parameters"""
    # Existing parameters
    initial_lr: float
    initial_grad_clip: float
    min_lr: float = 0.000001  # Lower minimum to allow more fine-tuning
    max_lr: float = 1.0
    min_grad_clip: float = 1e-3
    max_grad_clip: float = 100.0
    patience: int = 3
    grad_clip_decay: float = 0.95
    
    # New parameters
    initial_batch_size: int = 1024
    min_batch_size: int = 32
    max_batch_size: int = 4096
    batch_size_step: int = 32
    
    initial_momentum: float = 0.9
    min_momentum: float = 0.8
    max_momentum: float = 0.999
    
    initial_weight_decay: float = 0.00001
    min_weight_decay: float = 0.00000001
    max_weight_decay: float = 0.01
    
    def __post_init__(self):
        """Validate all configuration parameters"""
        if not 0 < self.initial_lr < 1:
            raise ValueError(f"Initial learning rate must be between 0 and 1, got {self.initial_lr}")
        if not self.min_batch_size <= self.initial_batch_size <= self.max_batch_size:
            raise ValueError(f"Invalid batch size range: {self.initial_batch_size}")
        if not self.min_momentum <= self.initial_momentum <= self.max_momentum:
            raise ValueError(f"Invalid momentum range: {self.initial_momentum}")

class AutoTuner:
    def __init__(self, optimizer, config: AutoTunerConfig):
        self.optimizer = optimizer
        self.config = config
        self.lr = config.initial_lr  # Always start with initial LR
        self.grad_clip = config.initial_grad_clip
        self.best_loss = float('inf')
        self.patience = config.patience
        self.wait = 0
        self.min_lr = config.min_lr
        self.max_lr = config.max_lr
        self.history = []
        self.momentum_window = 7  # Increased from 5 to require more history
        self.improvement_threshold = 0.01
        self.min_validations_before_tuning = 2  # Don't start tuning until we have enough data
        self.oscillation_count_threshold = 3  # Require multiple oscillations before adjusting
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        print(f"\033[94m[AutoTuner] Starting with initial learning rate: {self.lr:.6f}\033[0m")
    
    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing the current state."""
        return {
            'lr': self.lr,
            'grad_clip': self.grad_clip,
            'best_loss': self.best_loss,
            'wait': self.wait,
            'history': self.history,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from a dictionary, but keep initial learning rate."""
        # Don't restore learning rate, keep initial value
        self.grad_clip = state_dict.get('grad_clip', self.config.initial_grad_clip)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.wait = state_dict.get('wait', 0)
        self.history = state_dict.get('history', [])
        
        # Ensure optimizer has initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.initial_lr
        
    def step(self, val_loss):
        self.history.append(val_loss)
        
        # Check if this is the best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            return True
            
        self.wait += 1
        
        # Store old values for comparison
        old_lr = self.lr
        
        # Analyze loss trend - only if we have enough validation history
        if len(self.history) >= self.min_validations_before_tuning:
            if len(self.history) >= self.momentum_window:
                recent_trend = self.analyze_trend()
                
                if recent_trend == 'plateau':
                    self.lr *= 0.8  # Reduce learning rate
                    print(f"\033[93m[AutoTuner] Plateau detected - Reducing learning rate: {old_lr:.6f} -> {self.lr:.6f}\033[0m")
                    self.wait = 0
                elif recent_trend == 'oscillating':
                    self.lr *= 0.7  # Smooth descent with smaller steps
                    print(f"\033[93m[AutoTuner] Oscillation detected - Adjusting learning rate: {old_lr:.6f} -> {self.lr:.6f}\033[0m")
                elif recent_trend == 'improving':
                    self.lr *= 1.1  # Slightly increase learning rate
                    print(f"\033[92m[AutoTuner] Steady improvement - Increasing learning rate: {old_lr:.6f} -> {self.lr:.6f}\033[0m")
                
        # Ensure learning rate stays within bounds
        if self.lr != max(self.min_lr, min(self.max_lr, self.lr)):
            old_lr = self.lr
            self.lr = max(self.min_lr, min(self.max_lr, self.lr))
            print(f"\033[93m[AutoTuner] Learning rate clipped to bounds: {old_lr:.6f} -> {self.lr:.6f}\033[0m")
            
        # Update optimizer with new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            
        return False
        
    def analyze_trend(self):
        recent = self.history[-self.momentum_window:]
        
        # Calculate improvements
        improvements = [recent[i-1] - recent[i] for i in range(1, len(recent))]
        
        # Check for plateau
        if abs(sum(improvements)) < self.improvement_threshold:
            return 'plateau'
            
        # Check for oscillation - require multiple sign changes for more tolerance
        sign_changes = sum(1 for i, j in zip(improvements[:-1], improvements[1:]) if i * j < 0)
        if sign_changes >= self.oscillation_count_threshold:
            return 'oscillating'
            
        # Check if consistently improving
        if all(i > 0 for i in improvements):
            return 'improving'
            
        return 'normal'

def save_training_state():
    """Save the current training state including AutoTuner state"""
    train_track_dict = dict(train_track)
    train_track_dict['auto_tuner_state'] = auto_tuner.state_dict()
    miscfuncs.json_save(train_track_dict, 'training_stats', save_path)

def save_checkpoint(network, auto_tuner, train_track, epoch, save_path):
    # Create a simpler checkpoint filename
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}")
    
    # Save the checkpoint
    checkpoint = {
        'network_state': network.state_dict(),
        'auto_tuner_state': auto_tuner.state_dict(),
        'train_track': train_track
    }
    torch.save(checkpoint, checkpoint_path)

class DataShuffler:
    def __init__(self, dataset, train_ratio=0.7, val_ratio=0.15):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - (train_ratio + val_ratio)
        self.memory_limit = psutil.virtual_memory().total * 0.8  # 80% of available RAM
        self.batch_memory_threshold = self.memory_limit * 0.1    # 10% of memory limit
        
    def strategic_data_reorganization(self, data):
        # Get initial memory state
        initial_memory = psutil.Process().memory_info().rss
        batch_size = min(1000, len(data) // 10)  # Dynamic batch sizing
        
        # Process batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # Process batch operations here
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU
            
        # Get final memory state and calculate delta
        current_memory = psutil.Process().memory_info().rss
        memory_delta = (current_memory - initial_memory) / (1024 * 1024)  # Convert to MB
        
        if abs(memory_delta) > 0.01:  # Only log if change is significant
            print(f"\033[94m[DataShuffler] Memory delta: {memory_delta:.2f} MB")
            if memory_delta > 100:  # Warning threshold
                print(f"\033[93m[DataShuffler] Warning: Large memory increase detected!\033[0m")
        
        # Log peak memory usage for GPU if available
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            print(f"\033[94m[DataShuffler] Peak GPU memory: {peak_gpu_memory:.2f} MB\033[0m")
            
        return memory_delta

    def log_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print("\033[94m[Memory Usage Stats]")
        print(f"RSS: {memory_info.rss / (1024 * 1024):.2f} MB")
        print(f"VMS: {memory_info.vms / (1024 * 1024):.2f} MB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            cached = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"GPU Allocated: {allocated:.2f} MB")
            print(f"GPU Cached: {cached:.2f} MB")
        print("\033[0m")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class OptimizedLRManager:
    def __init__(self, optimizer, initial_lr=0.001):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            if self.patience_counter > 10:  # Reduce LR after 5 epochs without improvement
                self._reduce_lr()
            return False
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5
            print(f"Reducing LR to {param_group['lr']}")

class AdvancedLRManager:
    def __init__(self, optimizer, max_lr=0.01, min_lr=0.000001, total_epochs=100, 
                 warmup_epochs=5, cycles=3, patience=5):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cycles = cycles
        self.patience = patience
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.cycle_length = (total_epochs - warmup_epochs) // cycles
        
        # Early stopping parameters
        self.early_stop_threshold = patience * 2
        self.early_stop_counter = 0
        
        # Learning rate history
        self.lr_history = []
        
    def get_warmup_lr(self):
        """Linear warmup phase"""
        progress = self.current_epoch / self.warmup_epochs
        return self.min_lr + (self.max_lr - self.min_lr) * progress
    
    def get_cosine_lr(self):
        """Cosine annealing with warm restarts"""
        epoch_in_cycle = (self.current_epoch - self.warmup_epochs) % self.cycle_length
        progress = epoch_in_cycle / self.cycle_length
        cosine = math.cos(math.pi * progress)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cosine)
    
    def get_one_cycle_lr(self):
        """One Cycle Policy within each restart cycle"""
        epoch_in_cycle = (self.current_epoch - self.warmup_epochs) % self.cycle_length
        half_cycle = self.cycle_length // 2
        
        if epoch_in_cycle < half_cycle:
            # First half: linear increase
            progress = epoch_in_cycle / half_cycle
            return self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            # Second half: linear decrease
            progress = (epoch_in_cycle - half_cycle) / half_cycle
            return self.max_lr - (self.max_lr - self.min_lr) * progress
    
    def step(self, val_loss=None):
        """Main step function to be called after each epoch"""
        if val_loss is not None:
            self._handle_validation_loss(val_loss)
        
        # Determine the learning rate
        if self.current_epoch < self.warmup_epochs:
            lr = self.get_warmup_lr()
        else:
            # Combine One Cycle and Cosine Annealing
            one_cycle_lr = self.get_one_cycle_lr()
            cosine_lr = self.get_cosine_lr()
            lr = (one_cycle_lr + cosine_lr) / 2
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.lr_history.append(lr)
        self.current_epoch += 1
        
        return lr
    
    def _handle_validation_loss(self, val_loss):
        """Handle validation loss tracking and early stopping"""
        if val_loss < self.best_loss:
            print(f"\033[92m[LRManager] New best validation loss: {val_loss:.6f} (previous: {self.best_loss:.6f})\033[0m")
            self.best_loss = val_loss
            self.bad_epochs = 0
            self.early_stop_counter = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                old_max_lr = self.max_lr
                old_min_lr = self.min_lr
                # Reduce max_lr and min_lr when loss plateaus
                self.max_lr *= 0.5
                self.min_lr *= 0.5
                self.bad_epochs = 0
                print(f"\033[93m[LRManager] Loss plateau - Adjusting learning rate bounds:")
                print(f"  Max LR: {old_max_lr:.6f} -> {self.max_lr:.6f}")
                print(f"  Min LR: {old_min_lr:.6f} -> {self.min_lr:.6f}\033[0m")
            
            self.early_stop_counter += 1
    
    def should_stop(self):
        """Early stopping check"""
        return self.early_stop_counter >= self.early_stop_threshold
    
    def get_lr_stats(self):
        """Return learning rate statistics"""
        return {
            'current_lr': self.lr_history[-1] if self.lr_history else self.min_lr,
            'max_lr_used': max(self.lr_history) if self.lr_history else self.max_lr,
            'min_lr_used': min(self.lr_history) if self.lr_history else self.min_lr,
            'lr_history': self.lr_history
        }

def setup_logging(log_file: str = "training.log"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class TrainingConfig:
    # Model parameters
    model_type: str
    hidden_size: int
    num_layers: int
    
    # Training parameters
    batch_size: int
    learning_rate: float
    max_epochs: int
    
    # Audio parameters
    sample_rate: int
    audio_channels: int
    
    # Logging parameters
    log_interval: int
    save_interval: int
    
    def log_config(self):
        """Print and log all configuration parameters"""
        logger.info("\nTraining Configuration:")
        for field in fields(self):
            value = getattr(self, field.name)
            logger.info(f"{field.name}: {value}")

class CustomTrainingError(Exception):
    """Base exception class for training errors"""
    pass

class MemoryError(CustomTrainingError):
    """Raised when memory usage exceeds limits"""
    pass

class ValidationError(CustomTrainingError):
    """Raised when validation metrics indicate training issues"""
    pass

class PerformanceMonitor:
    """Monitors training performance metrics."""
    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()
        
    def update(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
    def get_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        return {
            'training_time': time.time() - self.start_time,
            'peak_memory': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'average_loss': np.mean(self.metrics['loss']) if self.metrics['loss'] else 0
        }

class AudioAugmenter:
    def __init__(self):
        self.sample_rate = 44100  # Standard audio rate
        
    def add_noise(self, signal, noise_level=0.005):
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
        
    def time_stretch(self, signal, rate=1.1):
        return librosa.effects.time_stretch(signal, rate=rate)
        
    def pitch_shift(self, signal, steps=2):
        return librosa.effects.pitch_shift(signal, sr=self.sample_rate, n_steps=steps)
        
    def apply_random_augmentation(self, signal):
        augmentations = [self.add_noise, self.time_stretch, self.pitch_shift]
        aug_func = random.choice(augmentations)
        return aug_func(signal)

class AudioPerformanceMonitor:
    def __init__(self):
        self.latency_history = []
        self.buffer_underruns = 0
        
    def measure_processing_time(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start
            self.latency_history.append(latency)
            if latency > 0.005:  # 5ms threshold
                logger.warning(f"High latency detected: {latency*1000:.2f}ms")
            return result
        return wrapper

class AudioLossFunctions:
    @staticmethod
    def spectral_loss(y_true, y_pred):
        """Compute loss in frequency domain"""
        stft_true = torch.stft(y_true, n_fft=2048, hop_length=512)
        stft_pred = torch.stft(y_pred, n_fft=2048, hop_length=512)
        return F.mse_loss(torch.abs(stft_true), torch.abs(stft_pred))
    
    @staticmethod
    def envelope_loss(y_true, y_pred):
        """Compare signal envelopes"""
        true_env = torch.abs(torch.nn.functional.avg_pool1d(
            y_true.abs(), kernel_size=512, stride=256))
        pred_env = torch.abs(torch.nn.functional.avg_pool1d(
            y_pred.abs(), kernel_size=512, stride=256))
        return F.l1_loss(true_env, pred_env)

class AudioBuffer:
    def __init__(self, buffer_size=2048):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
        
    def write(self, samples):
        """Write samples to buffer"""
        for sample in samples:
            self.buffer.append(sample)
            
    def read(self, num_samples):
        """Read samples from buffer"""
        if len(self.buffer) < num_samples:
            raise BufferUnderrunError("Not enough samples in buffer")
        return [self.buffer.popleft() for _ in range(num_samples)]

class AudioQualityMetrics:
    @staticmethod
    def signal_to_noise_ratio(clean, processed):
        noise = clean - processed
        return 20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(noise))
    
    @staticmethod
    def total_harmonic_distortion(signal, sample_rate):
        freqs, mags = scipy.signal.welch(signal, fs=sample_rate)
        fundamental = np.argmax(mags)
        harmonics = mags[fundamental*2:fundamental*10]
        return np.sum(harmonics) / mags[fundamental]

class RealTimeOptimizer:
    def __init__(self, model):
        self.model = model
        
    def optimize_for_realtime(self):
        """Optimize model for real-time processing"""
        self.model = torch.jit.script(self.model)  # JIT compilation
        self.model = self.model.half()  # Convert to FP16
        self.model.eval()  # Set to inference mode
        
    def process_chunk(self, audio_chunk):
        with torch.inference_mode():
            return self.model(audio_chunk)

class NeuralAudioFeatures:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
    def extract_features(self, audio):
        """Extract multiple audio features"""
        features = {
            'mel': self.mel_transform(audio),
            'mfcc': torchaudio.transforms.MFCC()(audio),
            'spectral_centroid': self.compute_spectral_centroid(audio),
            'zero_crossing_rate': self.compute_zcr(audio)
        }
        return features
        
    def compute_spectral_centroid(self, audio):
        """Compute spectral centroid (brightness)"""
        spectrogram = torch.stft(audio, n_fft=2048)
        magnitudes = torch.abs(spectrogram)
        freqs = torch.linspace(0, self.sample_rate/2, magnitudes.shape[1])
        return torch.sum(magnitudes * freqs) / torch.sum(magnitudes)

class AudioQualityAnalyzer:
    def __init__(self):
        self.metrics = {}
        
    def analyze_quality(self, original, processed):
        """Comprehensive audio quality analysis"""
        self.metrics = {
            'snr': self.signal_to_noise_ratio(original, processed),
            'thd': self.total_harmonic_distortion(processed),
            'spectral_flatness': self.spectral_flatness(processed),
            'crest_factor': self.crest_factor(processed),
            'phase_correlation': self.phase_correlation(original, processed)
        }
        return self.metrics
        
    def generate_report(self):
        """Generate detailed quality report"""
        report = "Audio Quality Analysis Report\n"
        report += "=" * 30 + "\n"
        for metric, value in self.metrics.items():
            report += f"{metric}: {value:.2f}\n"
        return report

class TrainingMonitor:
    def __init__(self, log_dir="training_logs"):
        self.writer = SummaryWriter(log_dir)
        self.metrics = defaultdict(list)
        
    def log_step(self, epoch, batch, **metrics):
        print(f"\nEpoch {epoch} | Batch {batch}")
        print("-" * 50)
        
        # Log basic training metrics
        print(f"Learning Rate: {metrics['lr']:.6f}")
        print(f"Training Loss: {metrics['train_loss']:.4f}")
        print(f"Validation Loss: {metrics.get('val_loss', 'N/A')}")
        
        # Log audio metrics
        print(f"SNR: {metrics.get('snr', 'N/A')}")
        print(f"THD: {metrics.get('thd', 'N/A')}")
        
        # Log system metrics
        print(f"Memory Usage: {metrics.get('memory', 'N/A')} MB")
        print(f"Batch Time: {metrics.get('batch_time', 'N/A')}s")
        
        # Save to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch * num_batches + batch)

class ParameterScheduler:
    def __init__(self):
        self.schedules = {
            'learning_rate': CosineAnnealingLR,
            'batch_size': LinearSchedule,
            'gradient_clip': ExponentialDecay
        }
    
    def update_parameters(self, epoch):
        """Update all scheduled parameters"""
        updates = {}
        for param_name, scheduler in self.schedules.items():
            updates[param_name] = scheduler.get_value(epoch)
        return updates

class EnhancedLogger:
    def log_training_step(self, epoch, batch, metrics):
        self.log.info(f"""
        Epoch {epoch} | Batch {batch}
        --------------------------------
        Learning Rate: {metrics['lr']:.6f}
        Gradient Norm: {metrics['grad_norm']:.4f}
        Training Loss: {metrics['train_loss']:.4f}
        Val Loss: {metrics['val_loss']:.4f}
        Memory Usage: {metrics['memory']:.2f} MB
        Audio Quality: {metrics['audio_metrics']}
        --------------------------------
        """)

class HyperParameterTracker:
    def __init__(self):
        self.params = {
            'optimization': {
                'learning_rate': [],
                'gradient_clip': [],
                'momentum': []
            },
            'training': {
                'batch_size': [],
                'epoch': [],
                'iterations': []
            },
            'audio': {
                'snr': [],
                'thd': [],
                'latency': []
            },
            'system': {
                'memory_usage': [],
                'gpu_utilization': [],
                'processing_time': []
            }
        }

class TrainingVisualizer:
    def __init__(self):
        self.writer = SummaryWriter('runs/experiment_1')
        
    def log_metrics(self, step, metrics_dict):
        for category, values in metrics_dict.items():
            for name, value in values.items():
                self.writer.add_scalar(f'{category}/{name}', value, step)
                
    def log_audio_samples(self, step, audio_dict):
        for name, (audio, sr) in audio_dict.items():
            self.writer.add_audio(name, audio, step, sample_rate=sr)

class TrainingAnalyzer:
    def analyze_training(self, metrics):
        analysis = {
            'convergence_rate': self.calculate_convergence_rate(metrics['loss']),
            'stability': self.assess_training_stability(metrics),
            'audio_quality': self.evaluate_audio_quality(metrics['audio']),
            'performance_bottlenecks': self.identify_bottlenecks(metrics['system'])
        }
        return analysis

class ComprehensiveMonitor:
    def __init__(self):
        self.logger = EnhancedLogger()
        self.visualizer = TrainingVisualizer()
        self.analyzer = TrainingAnalyzer()
        self.parameter_tracker = HyperParameterTracker()
        
    def log_training_step(self, epoch, batch, **metrics):
        # Log to console/file
        self.logger.log_training_step(epoch, batch, metrics)
        
        # Visualize in TensorBoard
        self.visualizer.log_metrics(epoch * num_batches + batch, metrics)
        
        # Track parameters
        self.parameter_tracker.update(metrics)
        
        # Periodic analysis
        if batch % analysis_interval == 0:
            analysis = self.analyzer.analyze_training(
                self.parameter_tracker.get_history()
            )
            self.logger.log_analysis(analysis)

class ProgressMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        
    def update(self, **metrics):
        current_time = time.time()
        metrics['time_elapsed'] = current_time - self.start_time
        
        # Update running metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
        # Print progress
        self._print_progress(metrics)
        
    def _print_progress(self, metrics):
        print("\nTraining Progress:")
        print(f"Time Elapsed: {metrics['time_elapsed']:.2f}s")
        print(f"Current Loss: {metrics['loss']:.4f}")
        print(f"Learning Rate: {metrics['lr']:.6f}")
        print(f"Audio Quality: SNR={metrics.get('snr', 'N/A')}")

class AudioMetricsTracker:
    def __init__(self):
        self.metrics = {
            'snr': [],
            'thd': [],
            'latency': [],
            'frequency_response': []
        }
    
    def compute_metrics(self, original, processed):
        metrics = {
            'snr': self._compute_snr(original, processed),
            'thd': self._compute_thd(processed),
            'latency': self._measure_latency(),
            'freq_response': self._analyze_frequency_response(processed)
        }
        self._update_metrics(metrics)
        return metrics

class GuitarAudioAugmenter:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def add_pickup_noise(self, signal, noise_level=0.001):
        """Simulate electromagnetic pickup noise"""
        hum_50hz = np.sin(2 * np.pi * 50 * np.arange(len(signal)) / self.sample_rate)
        hum_60hz = np.sin(2 * np.pi * 60 * np.arange(len(signal)) / self.sample_rate)
        noise = (hum_50hz + hum_60hz) * noise_level
        return signal + noise
        
    def add_pick_attack(self, signal, intensity=0.1):
        """Simulate pick attack transients"""
        attack = np.random.exponential(0.1, size=1000)
        attack = np.pad(attack, (0, len(signal)-1000))
        return signal * (1 + intensity * attack)
        
    def add_string_resonance(self, signal, fundamental_freq=82.41):  # E2 = 82.41 Hz
        """Add sympathetic string resonance"""
        harmonics = [1, 2, 3, 4, 5]  # First 5 harmonics
        resonance = np.zeros_like(signal)
        for h in harmonics:
            freq = fundamental_freq * h
            resonance += np.sin(2 * np.pi * freq * np.arange(len(signal)) / self.sample_rate)
        return signal + (resonance * 0.02)
        
    def add_amp_cabinet_ir(self, signal, ir_path="ir/cabinet.wav"):
        """Apply cabinet impulse response"""
        if os.path.exists(ir_path):
            ir, _ = librosa.load(ir_path, sr=self.sample_rate)
            return scipy.signal.convolve(signal, ir, mode='same')
        return signal
        
    def add_string_squeak(self, signal, num_squeaks=2):
        """Add random string squeak noises"""
        for _ in range(num_squeaks):
            pos = np.random.randint(0, len(signal))
            squeak_len = np.random.randint(500, 2000)
            squeak = np.random.normal(0, 0.1, squeak_len) * np.exp(-np.arange(squeak_len)/100)
            if pos + squeak_len < len(signal):
                signal[pos:pos+squeak_len] += squeak
        return signal
        
    def apply_random_augmentations(self, signal):
        """Apply random combination of augmentations"""
        augmentations = [
            (self.add_pickup_noise, {'noise_level': np.random.uniform(0.0005, 0.002)}),
            (self.add_pick_attack, {'intensity': np.random.uniform(0.05, 0.15)}),
            (self.add_string_resonance, {'fundamental_freq': np.random.choice([82.41, 110.0, 146.83])}),
            (self.add_string_squeak, {'num_squeaks': np.random.randint(0, 3)})
        ]
        
        # Apply 2-3 random augmentations
        num_augs = np.random.randint(2, 4)
        chosen_augs = random.sample(augmentations, num_augs)
        
        augmented = signal.copy()
        for aug_func, params in chosen_augs:
            augmented = aug_func(augmented, **params)
            
        return augmented

# Clean loss functions for accurate modeling
class EnhancedESRLoss(nn.Module):
    def __init__(self):
        super(EnhancedESRLoss, self).__init__()
        self.epsilon = 1e-6  # Slightly larger epsilon for numerical stability
        
    def forward(self, output, target):
        # Time-domain loss
        time_loss = torch.mean(torch.pow(output - target, 2))
        
        # Energy normalization
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        
        # Normalized loss
        return time_loss / energy

def train_epoch(network, dataset, optimizer, loss_fn, args):
    network.train()
    total_loss = 0
    
    # Process in chunks optimal for audio
    chunk_size = args.segment_length  # e.g., 22050 samples
    
    for i in range(0, len(dataset), chunk_size):
        # Get chunk of audio
        input_chunk = dataset[i:i+chunk_size]
        target_chunk = target[i:i+chunk_size]
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            output = network(input_chunk)
            loss = loss_fn(output, target_chunk)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(network.parameters(), args.gradient_clip)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (len(dataset) // chunk_size)

@torch.no_grad()
def validate(network, val_dataset, loss_fn, args):
    network.eval()
    total_loss = 0
    chunk_size = args.val_chunk  # e.g., 100000
    
    for i in range(0, len(val_dataset), chunk_size):
        input_chunk = val_dataset[i:i+chunk_size]
        target_chunk = val_target[i:i+chunk_size]
        
        output = network(input_chunk)
        loss = loss_fn(output, target_chunk)
        total_loss += loss.item()
        
    return total_loss / (len(val_dataset) // chunk_size)

# Add explicit memory cleanup
def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class CustomLogger:
    """Custom logging class for enhanced logging capabilities."""
    def __init__(self, log_file: str = "training.log"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message: str) -> None:
        """Log info level message."""
        self.logger.info(message)
    
    def error(self, message: str) -> None:
        """Log error level message."""
        self.logger.error(message)
    
    def warning(self, message: str) -> None:
        """Log warning level message."""
        self.logger.warning(message)

class EnhancedAutoTuner:
    """Enhanced automatic hyperparameter tuner."""
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        config: AutoTunerConfig,
        logger: CustomLogger
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize tuning parameters."""
        self.current_lr = self.config.initial_lr
        self.current_grad_clip = self.config.initial_grad_clip
        self.best_loss = float('inf')
        self.patience_counter = 0
        self._update_optimizer_params()
        
    def step(self, current_loss: float) -> bool:
        """
        Perform one step of parameter tuning.
        
        Args:
            current_loss: Current validation loss
            
        Returns:
            bool: True if parameters were updated
        """
        improved = False
        
        # Record statistics
        self.stats['losses'].append(current_loss)
        self.stats['learning_rates'].append(self.current_lr)
        
        if current_loss < self.best_loss:
            improved = True
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self._adjust_parameters()
                
        return improved
    
    def get_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return dict(self.stats)

class TrainingState:
    """Manages training state and checkpoints."""
    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
        self.best_model_path = os.path.join(save_path, 'best_model.pt')
        self.checkpoint_path = os.path.join(save_path, 'checkpoint.pt')
        
    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float
    ) -> None:
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_path)
        
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load training checkpoint."""
        if os.path.exists(self.checkpoint_path):
            return torch.load(self.checkpoint_path)
        return {}

class AsyncAudioProcessor:
    """Handles asynchronous audio processing with batch optimization."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 1024):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.processing_queue: List[torch.Tensor] = []
        self.results_cache: Dict[int, torch.Tensor] = {}
        
    async def process_batch(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Process audio data asynchronously in batches."""
        try:
            # Split data into batches
            batches = self._create_batches(audio_data)
            
            # Create tasks for each batch
            tasks = [
                self.executor.submit(self._process_single_batch, batch)
                for batch in batches
            ]
            
            # Gather results
            results = await asyncio.gather(
                *[asyncio.wrap_future(task) for task in tasks]
            )
            
            # Combine results
            return self._combine_results(results)
            
        except Exception as e:
            logger.error(f"Async processing error: {str(e)}")
            raise
            
    def _create_batches(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        """Split audio data into optimal batch sizes."""
        return torch.split(audio_data, self.batch_size)
        
    @staticmethod
    def _process_single_batch(batch: torch.Tensor) -> torch.Tensor:
        """Process a single batch of audio data."""
        # Add your processing logic here
        return processed_batch

class DistributedTrainer:
    """Handles distributed training across multiple GPUs/machines."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        world_size: int,
        rank: int
    ):
        self.world_size = world_size
        self.rank = rank
        self.model = model
        self.optimizer = optimizer
        
    def setup_distributed(self, backend: str = 'nccl') -> None:
        """Initialize distributed training environment."""
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Move model to GPU and wrap with DDP
        self.model = self.model.cuda(self.rank)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=True
        )
        
    def distributed_train_step(
        self,
        batch: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Execute a single distributed training step."""
        # Ensure data is on correct device
        batch = batch.cuda(self.rank)
        target = target.cuda(self.rank)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient synchronization
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
                
        self.optimizer.step()
        
        return {"loss": loss.item()}

class ModelProfiler:
    """Advanced model profiling for performance optimization."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
    def profile_model(
        self,
        sample_input: torch.Tensor,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Profile model performance and resource usage."""
        results = {}
        
        # Memory profiling
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        with self.profiler as prof:
            # Warmup
            for _ in range(3):
                self.model(sample_input)
                
            # Actual profiling
            start_time = time.perf_counter()
            output = self.model(sample_input)
            inference_time = time.perf_counter() - start_time
            
        # Collect results
        results['inference_time'] = inference_time
        results['memory_usage'] = torch.cuda.memory_allocated() - start_mem
        results['peak_memory'] = torch.cuda.max_memory_allocated()
        
        if detailed:
            results['layer_stats'] = self._analyze_layer_stats(prof)
            results['bottlenecks'] = self._identify_bottlenecks(prof)
            
        return results
        
    def _analyze_layer_stats(self, prof) -> Dict[str, List[Dict]]:
        """Analyze performance of individual layers."""
        return {
            'layer_times': prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            )
        }
        
    def _identify_bottlenecks(self, prof) -> List[Dict]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        for event in prof.key_averages():
            if event.cuda_time_total > 1000:  # microseconds
                bottlenecks.append({
                    'name': event.key,
                    'cuda_time': event.cuda_time_total,
                    'cpu_time': event.cpu_time_total,
                    'memory': event.cuda_memory_usage
                })
        return bottlenecks

@dataclass
class AudioTestConfig:
    """Configuration for audio testing parameters."""
    sample_rate: int = 44100
    min_snr: float = 20.0
    max_latency_ms: float = 5.0
    frequency_bands: List[tuple] = None
    
    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = [
                (20, 200),    # Low
                (200, 2000),  # Mid
                (2000, 20000) # High
            ]

class AudioModelTester:
    """Comprehensive audio model testing suite."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor,
        config: AudioTestConfig
    ):
        self.model = model
        self.test_data = test_data
        self.config = config
        self.results: Dict[str, Any] = {}
        
    def run_tests(self) -> Dict[str, Any]:
        """Run all audio quality and performance tests."""
        try:
            self.results.update(self.test_audio_quality())
            self.results.update(self.test_performance())
            self.results.update(self.test_memory_usage())
            self._validate_results()
            return self.results
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            raise
            
    def test_audio_quality(self) -> Dict[str, float]:
        """Test audio quality metrics."""
        with torch.no_grad():
            output = self.model(self.test_data)
            
        return {
            'snr': self._calculate_snr(output),
            'frequency_response': self._analyze_frequency_response(output),
            'thd': self._calculate_thd(output),
            'latency': self._measure_latency()
        }
        
    def _calculate_snr(self, output: torch.Tensor) -> float:
        """Calculate Signal-to-Noise Ratio."""
        noise = output - self.test_data
        signal_power = torch.mean(self.test_data ** 2)
        noise_power = torch.mean(noise ** 2)
        return 10 * torch.log10(signal_power / noise_power).item()

class ABTester:
    """Comparative testing between two model versions."""
    
    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        test_data: torch.Tensor,
        metrics: List[str]
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.test_data = test_data
        self.metrics = metrics
        self.results: Dict[str, Dict[str, float]] = {}
        
    def compare_models(self) -> Dict[str, Dict[str, Any]]:
        """Run comparative analysis between models."""
        # Test both models
        results_a = self.evaluate_model(self.model_a, "Model A")
        results_b = self.evaluate_model(self.model_b, "Model B")
        
        # Analyze differences
        comparison = self.analyze_differences(results_a, results_b)
        
        # Statistical significance testing
        significance = self.test_significance(results_a, results_b)
        
        return {
            "model_a": results_a,
            "model_b": results_b,
            "differences": comparison,
            "significance": significance
        }
        
    def evaluate_model(
        self,
        model: torch.nn.Module,
        name: str
    ) -> Dict[str, float]:
        """Evaluate a single model on all metrics."""
        results = {}
        
        with torch.no_grad():
            output = model(self.test_data)
            
            for metric in self.metrics:
                results[metric] = self._calculate_metric(
                    metric,
                    output,
                    self.test_data
                )
                
        return results
        
    def analyze_differences(
        self,
        results_a: Dict[str, float],
        results_b: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate percentage differences between models."""
        differences = {}
        
        for metric in self.metrics:
            diff = ((results_b[metric] - results_a[metric]) 
                   / results_a[metric] * 100)
            differences[metric] = diff
            
        return differences
        
    def test_significance(
        self,
        results_a: Dict[str, float],
        results_b: Dict[str, float]
    ) -> Dict[str, bool]:
        """Perform statistical significance testing."""
        from scipy import stats
        
        significance = {}
        for metric in self.metrics:
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                results_a[metric],
                results_b[metric]
            )
            significance[metric] = p_value < 0.05
            
        return significance

class AdaptiveValidator:
    def __init__(self, initial_freq: int = 5, min_freq: int = 2, max_freq: int = 20):
        self.current_freq = initial_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.loss_history = []
        self.stability_window = 5
        
    def should_validate(self, epoch: int) -> bool:
        """Determine if validation should be performed this epoch"""
        # Always validate first few epochs
        if epoch < 5:
            return True
            
        # Check if current epoch is due for validation
        return epoch % self.current_freq == 0
        
    def update_frequency(self, current_loss: float) -> None:
        """Update validation frequency based on loss trend"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) >= self.stability_window:
            old_freq = self.current_freq
            trend = self._analyze_trend()
            
            if trend == 'unstable':
                # More frequent validation when unstable
                self.current_freq = max(self.min_freq, self.current_freq - 1)
                if old_freq != self.current_freq:
                    print(f"\033[93m[Validator] Unstable training detected - Increasing validation frequency: {old_freq} -> {self.current_freq} epochs\033[0m")
            elif trend == 'stable':
                # Less frequent validation when stable
                self.current_freq = min(self.max_freq, self.current_freq + 2)
                if old_freq != self.current_freq:
                    print(f"\033[92m[Validator] Stable training detected - Decreasing validation frequency: {old_freq} -> {self.current_freq} epochs\033[0m")

    def _analyze_trend(self) -> str:
        """Analyze loss trend to determine training stability"""
        recent = self.loss_history[-self.stability_window:]

        # Calculate loss differences (improvements)
        differences = [recent[i] - recent[i-1] for i in range(1, len(recent))]

        # Calculate variance in loss to detect oscillations
        loss_variance = np.var(recent) if len(recent) > 1 else 0

        # Calculate average improvement
        avg_improvement = np.mean(differences) if differences else 0

        # Thresholds for stability detection
        variance_threshold = 0.001  # Low variance = stable
        improvement_threshold = 0.0001  # Minimal improvement = plateau

        # Determine trend
        if loss_variance > variance_threshold:
            return 'unstable'  # High variance indicates oscillations or instability
        elif abs(avg_improvement) < improvement_threshold:
            return 'stable'  # Low variance and minimal improvement = stable plateau
        else:
            return 'normal'  # Normal training progress

if __name__ == "__main__":
    """The main method creates the recurrent network, trains it and carries out validation/testing """
    start_time = time.time()

    # If a load_config argument was provided, construct the file path to the config file
    if args.load_config:
        # Load the configs and write them onto the args dictionary, this will add new args and/or overwrite old ones
        configs = miscfuncs.json_load(args.load_config, args.config_location)
        for parameters in configs:
            args.__setattr__(parameters, configs[parameters])

    if args.model == 'SimpleRNN':
        model_name = args.model + args.device + '_' + args.unit_type + '_hs' + str(args.hidden_size) + '_pre_' + args.pre_filt
    if args.pre_filt == 'A-Weighting':
        with open('Configs/' + 'b_Awght_mk2.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            args.pre_filt = list(reader)
            args.pre_filt = args.pre_filt[0]
            for item in range(len(args.pre_filt)):
                args.pre_filt[item] = float(args.pre_filt[item])
    elif args.pre_filt == 'high_pass':
        args.pre_filt = [-0.85, 1]
    elif args.pre_filt == 'None':
        args.pre_filt = None

    # Generate name of directory where results will be saved
    save_path = os.path.join(args.save_location, args.device + '-' + args.load_config)

    # Check if an existing saved model exists, and load it, otherwise creates a new model
    network = init_model(save_path, args)

    # Check if a cuda device is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize network, optimizer, and scheduler
    # Use lower beta2 for smoother updates (helps with high-frequency stability)
    optimiser = torch.optim.Adam(network.parameters(), 
                                lr=args.learn_rate,
                                weight_decay=args.weight_decay,
                                betas=(0.9, 0.99))  # Lower beta2 for smoother updates
    ### scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.90, patience=5)
    loss_functions = training.LossWrapper(args.loss_fcns, args.pre_filt)

    # Initialize train_track first
    train_track = training.TrainTrack()

    # Then handle restoration of previous stats
    if miscfuncs.file_check('training_stats.json', save_path):
        prev_stats = miscfuncs.json_load('training_stats', save_path)
        train_track.restore_data(prev_stats)
        config = AutoTunerConfig(
            initial_lr=args.learn_rate,  # Use default learning rate
            initial_grad_clip=args.gradient_clip,
            patience=10
        )
        auto_tuner = AutoTuner(optimiser, config=config)
        if 'auto_tuner_state' in prev_stats:
            auto_tuner.load_state_dict(prev_stats['auto_tuner_state'])
            print(f"\033[94m[AutoTuner] Restored previous best validation loss: {auto_tuner.best_loss:.6f}\033[0m")
        
        # Reset patience counter at start of training (using wait attribute)
        auto_tuner.wait = 0
    else:
        train_track['best_val_loss'] = float('inf')
        config = AutoTunerConfig(
            initial_lr=args.learn_rate,  # Use default learning rate
            initial_grad_clip=args.gradient_clip,
            patience=10
        )
        auto_tuner = AutoTuner(optimiser, config=config)

    writer = SummaryWriter(os.path.join('runs2', model_name))

    # Load dataset
    dataset = dataset.DataSet(data_dir=args.data_location)

    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(os.path.join('train', args.file_name), 'train')

    dataset.create_subset('val')
    dataset.load_file(os.path.join('val', args.file_name), 'val')

    dataset.create_subset('test')
    dataset.load_file(os.path.join('test', args.file_name), 'test')


    # If training is restarting, this will ensure the previously elapsed training time is added to the total
    init_time = time.time() - start_time + train_track['total_time']*3600
    # Set network save_state flag to true, so when the save_model method is called the network weights are saved
    network.save_state = True
    patience_counter = 0

    # Initialize at the start of training
    shuffler_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        # Add more config options as needed
    }

    data_shuffler = DataShuffler(dataset, **shuffler_config)

    # Initialize adaptive validator
    validator = AdaptiveValidator(
        initial_freq=args.validation_f,
        min_freq=2,
        max_freq=20
    )

    # Training loop
    try:
        for epoch in range(train_track['current_epoch'] + 1, args.epochs + 1):
            ep_st_time = time.time()

            # Run 1 epoch of training
            epoch_loss = network.train_epoch(dataset.subsets['train'].data['input'][0],
                                             dataset.subsets['train'].data['target'][0],
                                             loss_functions, optimiser, args.batch_size, args.init_len, args.up_fr)

            # Run validation
            if validator.should_validate(epoch):
                val_ep_st_time = time.time()
                val_output, val_loss = network.process_data(dataset.subsets['val'].data['input'][0],
                                                 dataset.subsets['val'].data['target'][0], 
                                                 loss_functions, args.val_chunk)
                
                # Use AutoTuner for all hyperparameter management
                improved = auto_tuner.step(val_loss.item())
                
                # Save checkpoint every N epochs
                if epoch % 10 == 0:
                    save_checkpoint(network, auto_tuner, train_track, epoch, save_path)
                
                # Apply the current gradient clipping value
                torch.nn.utils.clip_grad_norm_(network.parameters(), auto_tuner.grad_clip)
                
                current_lr = optimiser.param_groups[0]['lr']
                
                print(f"Validation loss: {val_loss.item():.6f}")

                
                if improved:
                    network.save_model('model_best', save_path)
                    write(os.path.join(save_path, "best_val_out.wav"),
                          dataset.subsets['test'].fs, val_output.cpu().numpy()[:, 0, 0])
                    print(f"\033[92mNew best validation loss: {val_loss.item():.6f}\033[0m")
                
                train_track.val_epoch_update(val_loss.item(), val_ep_st_time, time.time())
                writer.add_scalar('Loss/val', train_track['validation_losses'][-1], epoch)
                writer.add_scalar('LR/current', current_lr)

                # Update adaptive validator with current loss
                validator.update_frequency(val_loss.item())
                
                # Save training state including AutoTuner state
                train_track_dict = dict(train_track)
                train_track_dict['auto_tuner_state'] = auto_tuner.state_dict()
                miscfuncs.json_save(train_track_dict, 'training_stats', save_path)
                
                # Stop if learning rate becomes too small
                if current_lr < 1e-7:
                    print('Learning rate too small, stopping training')
                    break
                
                # Stop if validation hasn't improved for too long
                if args.validation_p and auto_tuner.wait > args.validation_p:
                    print('Validation patience limit reached')
                    break

            current_lr = optimiser.param_groups[0]['lr']
            print(f"current learning rate: {current_lr}")
            train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
            # write loss to the tensorboard (just for recording purposes)
            writer.add_scalar('Loss/train', train_track['training_losses'][-1], epoch)
            writer.add_scalar('LR/current', current_lr)
            network.save_model('model', save_path)
            save_training_state()

            if args.validation_p and patience_counter > args.validation_p:
                print('validation patience limit reached at epoch ' + str(epoch))
                break

            # Perform strategic data reorganization
            data_shuffler.strategic_data_reorganization(dataset.subsets['train'].data['input'][0])

    except Exception as e:
        print(f"\033[91mError during training: {str(e)}\033[0m")
        save_checkpoint(network, auto_tuner, train_track, epoch, save_path)
        raise e

    lossESR = training.ESRLoss()
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                     dataset.subsets['test'].data['target'][0], loss_functions, args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_final.wav"), dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 1)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 1)
    train_track['test_loss_final'] = test_loss.item()
    train_track['test_lossESR_final'] = test_loss_ESR.item()

    best_val_net = miscfuncs.json_load('model_best', save_path)
    network = networks.load_model(best_val_net)
    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                     dataset.subsets['test'].data['target'][0], loss_functions, args.test_chunk)
    test_loss_ESR = lossESR(test_output, dataset.subsets['test'].data['target'][0])
    write(os.path.join(save_path, "test_out_bestv.wav"),
          dataset.subsets['test'].fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 2)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 2)
    train_track['test_loss_best'] = test_loss.item()
    train_track['test_lossESR_best'] = test_loss_ESR.item()
    save_training_state()
    if torch.cuda.is_available():
        with open(os.path.join(save_path, 'maxmemusage.txt'), 'w') as f:
            f.write(str(torch.cuda.max_memory_allocated()))

