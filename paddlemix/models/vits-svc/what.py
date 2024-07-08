import paddle
import math
import numpy as np


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


if __name__ == "__main__":

    y = np.random.normal(loc=0.1406, scale=0.2395, size=[1, 1112384]).astype("float32")
    y_pd = paddle.to_tensor(y)

    n_fft = 1024
    hop_length = 320
    win_length = 1024

    window_pd = custom_hann_window_paddle(win_length)

    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = True
    return_complex = False

    spec_pd = paddle.signal.stft(
                y_pd, 
                n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window_pd, 
                center=center, 
                pad_mode='reflect', 
                normalized=False, 
                onesided=True)

    # spec_pd = paddle.randn([1, 513, 3474], dtype="float32") + 1j * paddle.randn([1, 513, 3474], dtype="float32")

    spec_pd_stack = paddle.stack( [spec_pd.real(), spec_pd.imag()], axis=-1 )
    spec_pd_as_real = paddle.as_real(spec_pd)

    print(
        paddle.abs(spec_pd_stack - spec_pd_as_real).max().item()
    )
