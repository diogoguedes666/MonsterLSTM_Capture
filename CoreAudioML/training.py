import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class SpectralLoss(nn.Module):
    """Spectral loss that emphasizes high frequency matching, especially above 10kHz"""
    def __init__(self, n_fft=1024, hop_length=512, sample_rate=44100, 
                 low_freq_weight=1.0, mid_freq_weight=2.0, high_freq_weight=5.0,
                 mid_cutoff=5000, high_cutoff=10000, excess_penalty=10.0):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.low_freq_weight = low_freq_weight
        self.mid_freq_weight = mid_freq_weight
        self.high_freq_weight = high_freq_weight
        self.mid_cutoff = mid_cutoff
        self.high_cutoff = high_cutoff
        self.excess_penalty = excess_penalty  # Extra penalty when output > target in high freqs
        self.epsilon = 1e-8
        
        # Create multi-band frequency weighting
        n_freq_bins = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freq_bins)
        # Create weighting with multiple bands
        self.register_buffer('freq_weights', torch.ones(n_freq_bins))
        mid_bin = int(mid_cutoff * n_freq_bins / (sample_rate / 2))
        high_bin = int(high_cutoff * n_freq_bins / (sample_rate / 2))
        
        # Smooth transition widths
        transition_width = n_freq_bins // 20
        
        for i in range(n_freq_bins):
            freq_hz = freqs[i].item()
            if freq_hz <= mid_cutoff:
                # Low frequencies: normal weight
                self.freq_weights[i] = self.low_freq_weight
            elif freq_hz <= high_cutoff:
                # Mid-high frequencies (5-10kHz): moderate weight with smooth transition
                progress = (freq_hz - mid_cutoff) / (high_cutoff - mid_cutoff)
                weight = self.low_freq_weight + (self.mid_freq_weight - self.low_freq_weight) * progress
                self.freq_weights[i] = weight
            else:
                # High frequencies (10kHz+): heavy weight with smooth transition
                # Exponential increase for very high frequencies
                progress = (freq_hz - high_cutoff) / ((sample_rate / 2) - high_cutoff)
                # Use exponential curve for more aggressive weighting at very high frequencies
                exp_progress = 1.0 - math.exp(-3.0 * progress)  # Exponential curve
                weight = self.mid_freq_weight + (self.high_freq_weight - self.mid_freq_weight) * exp_progress
                self.freq_weights[i] = weight

    def forward(self, output, target):
        """
        Compute spectral loss with frequency weighting
        Args:
            output: [seq_len, batch, channels]
            target: [seq_len, batch, channels]
        """
        # Handle different input shapes
        if output.dim() == 3:
            # Assuming [seq_len, batch, channels] format
            seq_len, batch_size, n_channels = output.shape
            
            # Convert to [batch, channels, seq_len] for easier processing
            output = output.permute(1, 2, 0)  # [batch, channels, seq_len]
            target = target.permute(1, 2, 0)  # [batch, channels, seq_len]
            
            total_loss = 0
            for b in range(batch_size):
                for ch in range(n_channels):
                    output_ch = output[b, ch, :]  # [seq_len]
                    target_ch = target[b, ch, :]  # [seq_len]
                    
                    # Compute STFT - torch.stft expects [batch, samples] or [samples]
                    # Add batch dimension if needed
                    if output_ch.dim() == 1:
                        output_ch = output_ch.unsqueeze(0)  # [1, seq_len]
                        target_ch = target_ch.unsqueeze(0)  # [1, seq_len]
                    
                    # Ensure sequence is long enough for STFT (at least n_fft samples)
                    seq_len_actual = output_ch.shape[-1]
                    if seq_len_actual < self.n_fft:
                        # Pad with zeros if sequence is too short
                        pad_size = self.n_fft - seq_len_actual
                        output_ch = torch.nn.functional.pad(output_ch, (0, pad_size), mode='constant', value=0.0)
                        target_ch = torch.nn.functional.pad(target_ch, (0, pad_size), mode='constant', value=0.0)
                    
                    output_stft = torch.stft(output_ch, n_fft=self.n_fft, hop_length=self.hop_length,
                                           return_complex=True, normalized=False)
                    target_stft = torch.stft(target_ch, n_fft=self.n_fft, hop_length=self.hop_length,
                                           return_complex=True, normalized=False)
                    
                    # Get magnitude spectra [n_freq, n_time]
                    output_mag = torch.abs(output_stft[0])  # Remove batch dim
                    target_mag = torch.abs(target_stft[0])  # Remove batch dim
                    
                    # Apply frequency weighting [n_freq, n_time]
                    freq_weights = self.freq_weights.unsqueeze(1)  # [n_freq, 1]
                    
                    # Standard weighted MSE
                    output_mag_weighted = output_mag * freq_weights
                    target_mag_weighted = target_mag * freq_weights
                    mag_diff = torch.pow(target_mag_weighted - output_mag_weighted, 2)
                    base_loss = torch.mean(mag_diff)
                    
                    # Asymmetric penalty: heavily penalize excess high frequencies (above 10kHz)
                    # This specifically targets when output has MORE energy than target
                    high_freq_bin = int(self.high_cutoff * self.freq_weights.shape[0] / (self.sample_rate / 2))
                    high_freq_mask = torch.zeros_like(self.freq_weights)
                    high_freq_mask[high_freq_bin:] = 1.0
                    high_freq_mask = high_freq_mask.unsqueeze(1)  # [n_freq, 1]
                    
                    # Find where output > target in high frequencies
                    excess_high_freq = torch.clamp(output_mag - target_mag, min=0.0) * high_freq_mask
                    # Apply extra penalty for excess high frequencies
                    excess_penalty_loss = self.excess_penalty * torch.mean(torch.pow(excess_high_freq, 2))
                    
                    # Combine losses
                    loss = base_loss + excess_penalty_loss
                    
                    # Normalize by target energy
                    target_energy = torch.mean(torch.pow(target_mag, 2)) + self.epsilon
                    loss = loss / target_energy
                    
                    total_loss += loss
            
            return total_loss / (batch_size * n_channels)
        else:
            raise ValueError(f"Unexpected input shape: {output.shape}")


# PreEmph is a class that applies an FIR pre-emphasis filter to the signal, the filter coefficients are in the
# filter_cfs argument, and lp is a flag that also applies a low pass filter
# Only supported for single-channel!
class PreEmph(nn.Module):
    def __init__(self, filter_cfs, low_pass=0):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = nn.Conv1d(1, 1, 2, bias=False)
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
        target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)

class LossWrapper(nn.Module):
    def __init__(self, losses, pre_filt=None):
        super(LossWrapper, self).__init__()
        loss_dict = {'ESR': ESRLoss(), 'DC': DCLoss(), 'Spectral': SpectralLoss()}
        if pre_filt:
            pre_filt = PreEmph(pre_filt)
            loss_dict['ESRPre'] = lambda output, target: loss_dict['ESR'].forward(*pre_filt(output, target))
        loss_functions = [[loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        loss = 0
        for i, losses in enumerate(self.loss_functions):
            loss += torch.mul(losses(output, target), self.loss_factors[i])
        return loss


class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0, 'training_losses': [], 'validation_losses': [], 'train_av_time': 0.0,
                     'val_av_time': 0.0, 'total_time': 0.0, 'best_val_loss': 1e12, 'test_loss': 0})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, loss, ep_st_time, ep_end_time, init_time, current_ep):
        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time
        self['training_losses'].append(loss)
        self['current_epoch'] = current_ep
        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, loss, ep_st_time, ep_end_time):
        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time
        self['validation_losses'].append(loss)
        if loss < self['best_val_loss']:
            self['best_val_loss'] = loss
