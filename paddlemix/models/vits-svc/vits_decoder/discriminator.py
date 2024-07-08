import torch
import paddle

from omegaconf import OmegaConf

try:
    from .msd import ScaleDiscriminator, ScaleDiscriminator_torch
    from .mpd import MultiPeriodDiscriminator, MultiPeriodDiscriminator_torch
    from .mrd import MultiResolutionDiscriminator, MultiResolutionDiscriminator_torch
except ImportError:
    import os, sys
    sys.path.append(
        os.path.expanduser("~/Desktop/PaddleMIX/paddlemix/models/vits-svc")
    )
    from vits_decoder.msd import ScaleDiscriminator, ScaleDiscriminator_torch
    from vits_decoder.mpd import MultiPeriodDiscriminator, MultiPeriodDiscriminator_torch
    from vits_decoder.mrd import MultiResolutionDiscriminator, MultiResolutionDiscriminator_torch


class Discriminator_torch(torch.nn.Module):
    def __init__(self, hp):
        super(Discriminator_torch, self).__init__()
        self.MRD = MultiResolutionDiscriminator_torch(hp)
        self.MPD = MultiPeriodDiscriminator_torch(hp)
        self.MSD = ScaleDiscriminator_torch()

    def forward(self, x):
        r = self.MRD(x)
        p = self.MPD(x)
        s = self.MSD(x)
        return r + p + s


class Discriminator(paddle.nn.Layer):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)
        self.MSD = ScaleDiscriminator()

    def forward(self, x):
        r = self.MRD(x)
        p = self.MPD(x)
        s = self.MSD(x)
        return r + p + s






if __name__ == '__main__':
    hp = OmegaConf.load('configs/base.yaml')
    model = Discriminator_torch(hp)

    x = torch.randn(3, 1, 16384)
    print(x.shape)

    output = model(x)
    for features, score in output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


if __name__ == '__main__':
    hp = OmegaConf.load('configs/base.yaml')
    model = Discriminator(hp)

    x = paddle.randn([3, 1, 16384])
    print(x.shape)

    output = model(x)
    for features, score in output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    paddle_total_params = sum(p.numel()
                               for p in model.parameters() if not p.stop_gradient)
    print(paddle_total_params.cpu().item())
