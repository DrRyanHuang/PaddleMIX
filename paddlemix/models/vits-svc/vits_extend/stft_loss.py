# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import math
import paddle
import numpy as np
import torch
import torch.nn.functional as F


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



def stft_torch(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = paddle.signal.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft.real()
    imag = x_stft.imag()

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return paddle.sqrt(paddle.clip(real ** 2 + imag ** 2, min=1e-7)).transpose([0, 2, 1])


if __name__ == "__main__":
    fft_size, hop_size, win_length = 1024, 120, 600
    x = np.random.rand(8, 8000)
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    window = np.random.rand(600)
    window_tc = torch.from_numpy(window).cuda()
    window_pd = paddle.to_tensor(window)

    y_tc = stft_torch(x_tc, fft_size, hop_size, win_length, window_tc).detach().cpu().numpy()
    y_pd = stft(x_pd, fft_size, hop_size, win_length, window_pd).detach().cpu().numpy()

    print(
        "stft:\n",
        (y_tc - y_pd).max().item(),
        y_tc.mean() - y_pd.mean(),
        y_tc.std() - y_pd.std(),
    )




class SpectralConvergengeLoss_torch(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss_torch, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class SpectralConvergengeLoss(paddle.nn.Layer):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return paddle.norm(y_mag - x_mag, p="fro") / paddle.norm(y_mag, p="fro")


if __name__ == "__main__":

    x = np.random.rand(8, 67, 513)
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    x = np.random.rand(8, 67, 513)
    x_tc1 = torch.from_numpy(x).cuda()
    x_pd1 = paddle.to_tensor(x)

    model_tc = SpectralConvergengeLoss_torch()
    model_pd = SpectralConvergengeLoss()

    y_tc = model_tc(x_tc, x_tc1).detach().cpu().numpy()
    y_pd = model_pd(x_pd, x_pd1).detach().cpu().numpy()

    print(
        "\nSpectralConvergengeLoss:\n",
        (y_tc - y_pd).max().item(),
        y_tc.mean() - y_pd.mean(),
        y_tc.std() - y_pd.std(),
    )





class LogSTFTMagnitudeLoss_torch(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss_torch, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class LogSTFTMagnitudeLoss(paddle.nn.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return paddle.nn.functional.l1_loss(paddle.log(y_mag), paddle.log(x_mag))


if __name__ == "__main__":

    x = np.random.rand(8, 67, 513)
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    x = np.random.rand(8, 67, 513)
    x_tc1 = torch.from_numpy(x).cuda()
    x_pd1 = paddle.to_tensor(x)

    model_tc = LogSTFTMagnitudeLoss_torch()
    model_pd = LogSTFTMagnitudeLoss()

    y_tc = model_tc(x_tc, x_tc1).detach().cpu().numpy()
    y_pd = model_pd(x_pd, x_pd1).detach().cpu().numpy()

    print(
        "\nLogSTFTMagnitudeLoss:\n",
        (y_tc - y_pd).max().item(),
        y_tc.mean() - y_pd.mean(),
        y_tc.std() - y_pd.std(),
    )




class STFTLoss_torch(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss_torch, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length).cuda()
        self.spectral_convergenge_loss = SpectralConvergengeLoss_torch()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss_torch()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft_torch(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft_torch(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss(paddle.nn.Layer):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = custom_hann_window_paddle(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


if __name__ == "__main__":

    fft_size, shift_size, win_length, window = 1024, 120, 600, 'hann_window'

    model_tc = STFTLoss_torch(fft_size, shift_size, win_length, window)
    model_pd = STFTLoss(fft_size, shift_size, win_length, window)

    x_np = np.random.rand(8, 8000).astype("float32")
    y_np = np.random.rand(8, 8000).astype("float32")

    x_tc = torch.from_numpy(x_np).cuda()
    y_tc = torch.from_numpy(y_np).cuda()

    x_pd = paddle.to_tensor(x_np)
    y_pd = paddle.to_tensor(y_np)

    out_pd1, out_pd2 = model_tc(x_tc, y_tc)
    out_tc1, out_tc2 = model_pd(x_pd, y_pd)

    out_pd1 = out_pd1.detach().cpu().numpy()
    out_pd2 = out_pd2.detach().cpu().numpy()
    out_tc1 = out_tc1.detach().cpu().numpy()
    out_tc2 = out_tc2.detach().cpu().numpy()

    print(
        "\nSTFTLoss out1:\n",
        (out_tc1 - out_pd1).max().item(),
        out_tc1.mean() - out_pd1.mean(),
        out_tc1.std() - out_pd1.std(),
    )

    print(
        "\nSTFTLoss out2:\n",
        (out_pd2 - out_tc2).max().item(),
        out_pd2.mean() - out_tc2.mean(),
        out_pd2.std() - out_tc2.std(),
    )






class MultiResolutionSTFTLoss_torch(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 resolutions,
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss_torch, self).__init__()
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss_torch(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss




class MultiResolutionSTFTLoss(paddle.nn.Layer):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 resolutions,
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = paddle.nn.LayerList()
        for fs, ss, wl in resolutions:
            self.stft_losses.append(
                STFTLoss(fs, ss, wl, window)
            )

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss




if __name__ == "__main__":

    window = 'hann_window'
    resolutions = [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]

    model_tc = MultiResolutionSTFTLoss_torch(resolutions, window)
    model_pd = MultiResolutionSTFTLoss(resolutions, window)

    x_np = np.random.rand(8, 8000).astype("float32")
    y_np = np.random.rand(8, 8000).astype("float32")

    x_tc = torch.from_numpy(x_np).cuda()
    y_tc = torch.from_numpy(y_np).cuda()

    x_pd = paddle.to_tensor(x_np)
    y_pd = paddle.to_tensor(y_np)

    out_pd1, out_pd2 = model_tc(x_tc, y_tc)
    out_tc1, out_tc2 = model_pd(x_pd, y_pd)

    out_pd1 = out_pd1.detach().cpu().numpy()
    out_pd2 = out_pd2.detach().cpu().numpy()
    out_tc1 = out_tc1.detach().cpu().numpy()
    out_tc2 = out_tc2.detach().cpu().numpy()

    print(
        "\nMultiResolutionSTFTLoss out1:\n",
        (out_tc1 - out_pd1).max().item(),
        out_tc1.mean() - out_pd1.mean(),
        out_tc1.std() - out_pd1.std(),
    )

    print(
        "\nMultiResolutionSTFTLoss out2:\n",
        (out_pd2 - out_tc2).max().item(),
        out_pd2.mean() - out_tc2.mean(),
        out_pd2.std() - out_tc2.std(),
    )


