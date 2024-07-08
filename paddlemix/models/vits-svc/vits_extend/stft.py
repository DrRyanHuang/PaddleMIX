# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import random
import torch
import paddle
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn


def custom_hann_window_paddle(window_length, periodic=True, dtype=None,):
    if dtype is None:
        dtype = 'float32'
    if periodic:
        window_length += 1
    n = paddle.arange(dtype=dtype, end=window_length)
    window = 0.5 - 0.5 * paddle.cos(x=2 * math.pi * n / (window_length - 1))
    if periodic:
        window = window[:-1]
    return window



class TacotronSTFT_torch(torch.nn.Module):
    def __init__(self, filter_length=512, hop_length=160, win_length=512,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=None, center=False):
        super(TacotronSTFT_torch, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)

        mel_basis = torch.from_numpy(mel).float().cuda()
        hann_window = torch.hann_window(win_length).cuda()

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    def linear_spectrogram(self, y):
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        y = torch.nn.functional.pad(y.unsqueeze(1),
                                    (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
                                    mode='reflect')
        y = y.squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.norm(spec, p=2, dim=-1)

        return spec

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        y = torch.nn.functional.pad(y.unsqueeze(1),
                                    (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = self.spectral_normalize_torch(spec)

        return spec

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)





class TacotronSTFT(paddle.nn.Layer):

    def __init__(self, filter_length=512, hop_length=160, win_length=512,
        n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0, mel_fmax=None,
        center=False):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=
            n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = paddle.to_tensor(data=mel).astype(dtype='float32')
            
        hann_window = custom_hann_window_paddle(win_length)
        self.register_buffer(name='mel_basis', tensor=mel_basis)
        self.register_buffer(name='hann_window', tensor=hann_window)

    def linear_spectrogram(self, y):
        assert paddle.min(x=y.data) >= -1
        assert paddle.max(x=y.data) <= 1

        y = paddle.nn.functional.pad(
            y.unsqueeze(axis=1), 
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)), 
            mode='reflect', 
            data_format='NCL')


        y = y.squeeze(axis=1)
        spec = paddle.signal.stft(y, self.n_fft, hop_length=self.hop_size,
            win_length=self.win_size, window=self.hann_window, center=self.
            center, pad_mode='reflect', normalized=False, onesided=True)
        spec = paddle.stack( [spec.real(), spec.imag()], axis=-1 )

        spec = paddle.linalg.norm(x=spec, p=2, axis=-1)
        return spec

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert paddle.min(x=y.data) >= -1
        assert paddle.max(x=y.data) <= 1

        y = paddle.nn.functional.pad(
            y.unsqueeze(axis=1), 
            (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)), 
            mode='reflect', 
            data_format='NCL')
        
        y = y.squeeze(axis=1)
        spec = paddle.signal.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                          center=self.center, pad_mode='reflect', normalized=False, onesided=True)
        spec = paddle.stack( [spec.real(), spec.imag()], axis=-1 )

        spec = paddle.sqrt(x=spec.pow(y=2).sum(axis=-1) + 1e-09)
        spec = paddle.matmul(x=self.mel_basis, y=spec)
        spec = self.spectral_normalize_torch(spec)
        return spec

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-05):
        return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)



if __name__ == "__main__":

    x = np.random.rand(8, 128000).astype("float32")
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.from_numpy(x).cuda()

    tc_param = 1024, 320, 1024, 100, 32000, 50.0, 16000.0, False
    pd_param = 1024, 320, 1024, 100, 32000, 50.0, 16000.0, False

    model_tc = TacotronSTFT_torch(*tc_param)
    model_pd = TacotronSTFT(*pd_param)

    y_tc = model_tc.mel_spectrogram(x_torch).detach().cpu().numpy()
    y_pd = model_pd.mel_spectrogram(x_paddle).detach().cpu().numpy()

    print(
        "mel_spectrogram\n",
        (y_tc - y_pd).max().item(),
        y_tc.mean() - y_pd.mean(),
        y_tc.std() - y_pd.std(),
    )

    y_tc = model_tc.linear_spectrogram(x_torch).detach().cpu().numpy()
    y_pd = model_pd.linear_spectrogram(x_paddle).detach().cpu().numpy()

    print(
        "linear_spectrogram\n",
        (y_tc - y_pd).max().item(),
        y_tc.mean() - y_pd.mean(),
        y_tc.std() - y_pd.std(),
    )