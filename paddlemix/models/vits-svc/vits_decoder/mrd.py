import torch
import torch.nn as nn   
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import paddle



class DiscriminatorR_torch(torch.nn.Module):
    def __init__(self, hp=None, 
                 resolution=(1024, 120, 600), 
                 lReLU_slope=0.2, 
                 use_spectral_norm=False
):
        super(DiscriminatorR_torch, self).__init__()

        self.resolution = resolution

        if hp is None:
            self.LRELU_SLOPE = lReLU_slope
            norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        else:
            self.LRELU_SLOPE = hp.mpd.lReLU_slope
            norm_f = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)

            break
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')

        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=False) #[B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag





class DiscriminatorR(paddle.nn.Layer):
    def __init__(self, 
                 hp=None, 
                 resolution=(1024, 120, 600), 
                 lReLU_slope=0.2, 
                 use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = lReLU_slope

        norm_f = paddle.nn.utils.weight_norm if use_spectral_norm == False else paddle.nn.utils.spectral_norm

        self.convs = paddle.nn.LayerList([
            norm_f(paddle.nn.Conv2D(1, 32, (3, 9), padding=(1, 4))),
            norm_f(paddle.nn.Conv2D(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(paddle.nn.Conv2D(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(paddle.nn.Conv2D(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(paddle.nn.Conv2D(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(paddle.nn.Conv2D(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
            break
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = paddle.nn.functional.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect', data_format="NCL")
        x = x.squeeze(1)
        x = paddle.signal.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False) # [B, F, TT, 2]
        x = paddle.stack( [x.real(), x.imag()], axis=-1 )
        mag = paddle.norm(x, p=2, axis =-1) #[B, F, TT]

        return mag




def DiscriminatorR_torch2paddle(torch_model, paddle_model):

    for i in range(5):
        
        paddle_model.convs[i].weight_v.set_value(
            paddle.to_tensor( torch_model.convs[i].weight_v.detach().cpu().numpy() )
        )
        paddle_model.convs[i].weight_g.set_value(
            paddle.to_tensor( torch_model.convs[i].weight_g.detach().cpu().numpy().squeeze() )
        )
        paddle_model.convs[i].bias.set_value(
            paddle.to_tensor( torch_model.convs[i].bias.detach().cpu().numpy() )
        )
    
    paddle_model.conv_post.weight_v.set_value(
        paddle.to_tensor( torch_model.conv_post.weight_v.detach().cpu().numpy() )
    )
    paddle_model.conv_post.weight_g.set_value(
        paddle.to_tensor( torch_model.conv_post.weight_g.detach().cpu().numpy()[0,0,0] )
    )
    paddle_model.conv_post.bias.set_value(
        paddle.to_tensor( torch_model.conv_post.bias.detach().cpu().numpy() )
    )


if __name__ == "__main__":

    torch_model = DiscriminatorR_torch().cuda()
    paddle_model = DiscriminatorR()

    import numpy as np
    x_np = np.random.rand(8, 1, 8000).astype("float32")
    x_tc = torch.from_numpy(x_np).cuda()
    x_pd = paddle.to_tensor(x_np)

    DiscriminatorR_torch2paddle(torch_model, paddle_model)

    _, y_tc = torch_model(x_tc)
    _, y_pd = paddle_model(x_pd)

    y_tc = y_tc.detach().cpu().numpy()
    y_pd = y_pd.detach().cpu().numpy()

    print(
        "DiscriminatorR",
        (y_tc - y_pd).max().item()
    )






class MultiResolutionDiscriminator_torch(torch.nn.Module):
    def __init__(self, 
                 hp=None, 
                 resolutions=[(1024, 120, 600), 
                              (2048, 240, 1200), 
                              (4096, 480, 2400), 
                              (512, 50, 240)]):
        super(MultiResolutionDiscriminator_torch, self).__init__()
        if hp is None:
            self.resolutions = resolutions
        else:
            self.resolutions = eval(hp.mrd.resolutions) 
        self.discriminators = nn.ModuleList(
            [DiscriminatorR_torch(hp, resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]


class MultiResolutionDiscriminator(paddle.nn.Layer):
    def __init__(self, 
                 hp=None, 
                 resolutions=[(1024, 120, 600), 
                              (2048, 240, 1200), 
                              (4096, 480, 2400), 
                              (512, 50, 240)]):
        super(MultiResolutionDiscriminator, self).__init__()
        if hp is None:
            self.resolutions = resolutions
        else:
            self.resolutions = eval(hp.mrd.resolutions) 
        self.discriminators = paddle.nn.LayerList(
            [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]



if __name__ == "__main__":

    torch_model = MultiResolutionDiscriminator_torch().cuda()
    paddle_model = MultiResolutionDiscriminator()

    import numpy as np
    x_np = np.random.rand(8, 1, 8000).astype("float32")
    x_tc = torch.from_numpy(x_np).cuda()
    x_pd = paddle.to_tensor(x_np)

    for i in range(4):
        DiscriminatorR_torch2paddle(torch_model.discriminators[i], 
                                    paddle_model.discriminators[i])

    y_tc = torch_model(x_tc)
    y_pd = paddle_model(x_pd)


    for i in range(4):

        _, _y_tc = y_tc[i]
        _, _y_pd = y_pd[i]

        _y_tc = _y_tc.detach().cpu().numpy()
        _y_pd = _y_pd.detach().cpu().numpy()

        print(
            "MultiResolutionDiscriminator",
            (_y_tc - _y_pd).max().item()
        )
