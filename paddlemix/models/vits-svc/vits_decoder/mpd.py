import torch
import paddle
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm




class DiscriminatorP_torch(torch.nn.Module):
    def __init__(self, hp=None, period=2, lReLU_slope=0.2, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP_torch, self).__init__()

        self.period = period

        if hp is None:
            self.LRELU_SLOPE = lReLU_slope
            kernel_size = kernel_size
            stride = stride
            norm_f = torch.nn.utils.weight_norm if use_spectral_norm == False else torch.nn.utils.spectral_norm
        else:
            self.LRELU_SLOPE = hp.mpd.lReLU_slope
            kernel_size = hp.mpd.kernel_size
            stride = hp.mpd.stride
            norm_f = torch.nn.utils.weight_norm if hp.mpd.use_spectral_norm == False else torch.nn.utils.spectral_norm

        self.convs = torch.nn.ModuleList([
            norm_f(torch.nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(torch.nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(torch.nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(torch.nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(torch.nn.Conv2d(512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x



class DiscriminatorP(paddle.nn.Layer):
    def __init__(self, hp=None, period=2, lReLU_slope=0.2, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()

        self.period = period

        if hp is None:
            self.LRELU_SLOPE = lReLU_slope
            kernel_size = kernel_size
            stride = stride
            norm_f = paddle.nn.utils.weight_norm if use_spectral_norm == False else paddle.nn.utils.spectral_norm
        else:
            self.LRELU_SLOPE = hp.mpd.lReLU_slope
            kernel_size = hp.mpd.kernel_size
            stride = hp.mpd.stride
            norm_f = paddle.nn.utils.weight_norm if hp.mpd.use_spectral_norm == False else paddle.nn.utils.spectral_norm

        self.convs = paddle.nn.LayerList([
            norm_f(paddle.nn.Conv2D(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(paddle.nn.Conv2D(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(paddle.nn.Conv2D(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(paddle.nn.Conv2D(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(paddle.nn.Conv2D(512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm_f(paddle.nn.Conv2D(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = paddle.nn.functional.pad(x, (0, n_pad), "reflect", data_format="NCL")
            t = t + n_pad
        x = x.reshape([b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x, 1, -1)

        return fmap, x





def DiscriminatorP_torch2paddle(torch_model, paddle_model):

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

    torch_model = DiscriminatorP_torch().cuda()
    paddle_model = DiscriminatorP()

    import numpy as np
    x_np = np.random.rand(8, 1, 8000).astype("float32")
    x_tc = torch.from_numpy(x_np).cuda()
    x_pd = paddle.to_tensor(x_np)

    for i in range(5):
        DiscriminatorP_torch2paddle(torch_model, paddle_model)

    _, y_tc = torch_model(x_tc)
    _, y_pd = paddle_model(x_pd)

    y_tc = y_tc.detach().cpu().numpy()
    y_pd = y_pd.detach().cpu().numpy()

    print(
        "DiscriminatorR",
        (y_tc - y_pd).max().item()
    )



class MultiPeriodDiscriminator_torch(torch.nn.Module):
    def __init__(self, hp=None, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator_torch, self).__init__()
        if hp is None:
            self.discriminators = torch.nn.ModuleList(
                [DiscriminatorP_torch(hp, period) for period in periods]
            )
        else:
            self.discriminators = torch.nn.ModuleList(
                [DiscriminatorP_torch(hp, period) for period in hp.mpd.periods]
            )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]


class MultiPeriodDiscriminator(paddle.nn.Layer):
    def __init__(self, hp=None, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()

        if hp is None:
            self.discriminators = paddle.nn.LayerList(
                [DiscriminatorP(hp, period) for period in periods]
            )
        else:
            self.discriminators = paddle.nn.LayerList(
                [DiscriminatorP(hp, period) for period in hp.mpd.periods]
            )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]


if __name__ == "__main__":

    torch_model = MultiPeriodDiscriminator_torch().cuda()
    paddle_model = MultiPeriodDiscriminator()

    import numpy as np
    x_np = np.random.rand(8, 1, 8000).astype("float32")
    x_tc = torch.from_numpy(x_np).cuda()
    x_pd = paddle.to_tensor(x_np)

    for i in range(5):
        DiscriminatorP_torch2paddle(torch_model.discriminators[i], 
                                    paddle_model.discriminators[i])

    for i in range(5):
        _, y_tc = torch_model(x_tc)[i]
        _, y_pd = paddle_model(x_pd)[i]

        y_tc = y_tc.detach().cpu().numpy()
        y_pd = y_pd.detach().cpu().numpy()

        print(
            "MultiPeriodDiscriminator",
            (y_tc - y_pd).max().item()
        )