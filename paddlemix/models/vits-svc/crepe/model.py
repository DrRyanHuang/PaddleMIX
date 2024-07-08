import functools

import numpy as np
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import crepe
# class x:
#     PITCH_BINS = 360
# crepe = x()


###########################################################################
# Model definition
###########################################################################


class Crepe(paddle.nn.Layer):
    """Crepe model definition"""

    def __init__(self, model='full'):
        super().__init__()

        # Model-specific layer parameters
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f'Model {model} is not supported')

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(nn.BatchNorm2D,
                                          epsilon=0.0010000000474974513,
                                          momentum=0.0)

        # Layer definitions
        self.conv1 = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = nn.Conv2D(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = nn.Conv2D(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.classifier = nn.Linear(
            in_features=self.in_features,
            out_features=crepe.PITCH_BINS)

    def forward(self, x, embed=False):
        import paddle.nn.functional as F
        # Forward pass through first five layers
        x = self.embed(x)

        if embed:
            return x

        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.transpose([0, 2, 1, 3]).reshape([-1, self.in_features])

        # Compute logits
        return F.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)

        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))


# import torch
# class Crepe_torch(torch.nn.Module):
#     """Crepe model definition"""

#     def __init__(self, model='full'):
#         super().__init__()

#         # Model-specific layer parameters
#         if model == 'full':
#             in_channels = [1, 1024, 128, 128, 128, 256]
#             out_channels = [1024, 128, 128, 128, 256, 512]
#             self.in_features = 2048
#         elif model == 'tiny':
#             in_channels = [1, 128, 16, 16, 16, 32]
#             out_channels = [128, 16, 16, 16, 32, 64]
#             self.in_features = 256
#         else:
#             raise ValueError(f'Model {model} is not supported')

#         # Shared layer parameters
#         kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
#         strides = [(4, 1)] + 5 * [(1, 1)]

#         # Overload with eps and momentum conversion given by MMdnn
#         batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
#                                           eps=0.0010000000474974513,
#                                           momentum=0.0)

#         # Layer definitions
#         self.conv1 = torch.nn.Conv2d(
#             in_channels=in_channels[0],
#             out_channels=out_channels[0],
#             kernel_size=kernel_sizes[0],
#             stride=strides[0])
#         self.conv1_BN = batch_norm_fn(
#             num_features=out_channels[0])

#         self.conv2 = torch.nn.Conv2d(
#             in_channels=in_channels[1],
#             out_channels=out_channels[1],
#             kernel_size=kernel_sizes[1],
#             stride=strides[1])
#         self.conv2_BN = batch_norm_fn(
#             num_features=out_channels[1])

#         self.conv3 = torch.nn.Conv2d(
#             in_channels=in_channels[2],
#             out_channels=out_channels[2],
#             kernel_size=kernel_sizes[2],
#             stride=strides[2])
#         self.conv3_BN = batch_norm_fn(
#             num_features=out_channels[2])

#         self.conv4 = torch.nn.Conv2d(
#             in_channels=in_channels[3],
#             out_channels=out_channels[3],
#             kernel_size=kernel_sizes[3],
#             stride=strides[3])
#         self.conv4_BN = batch_norm_fn(
#             num_features=out_channels[3])

#         self.conv5 = torch.nn.Conv2d(
#             in_channels=in_channels[4],
#             out_channels=out_channels[4],
#             kernel_size=kernel_sizes[4],
#             stride=strides[4])
#         self.conv5_BN = batch_norm_fn(
#             num_features=out_channels[4])

#         self.conv6 = torch.nn.Conv2d(
#             in_channels=in_channels[5],
#             out_channels=out_channels[5],
#             kernel_size=kernel_sizes[5],
#             stride=strides[5])
#         self.conv6_BN = batch_norm_fn(
#             num_features=out_channels[5])

#         self.classifier = torch.nn.Linear(
#             in_features=self.in_features,
#             out_features=crepe.PITCH_BINS)

#     def forward(self, x, embed=False):


#         # Forward pass through first five layers
#         x = self.embed(x)

#         if embed:
#             return x

#         # Forward pass through layer six
#         x = self.layer(x, self.conv6, self.conv6_BN)

#         # shape=(batch, self.in_features)
#         x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

#         # Compute logits
#         return torch.sigmoid(self.classifier(x))

#     ###########################################################################
#     # Forward pass utilities
#     ###########################################################################

#     def embed(self, x):
#         """Map input audio to pitch embedding"""
#         # shape=(batch, 1, 1024, 1)
#         x = x[:, None, :, None]

#         # Forward pass through first five layers
#         x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
#         x = self.layer(x, self.conv2, self.conv2_BN)
#         x = self.layer(x, self.conv3, self.conv3_BN)
#         x = self.layer(x, self.conv4, self.conv4_BN)
#         x = self.layer(x, self.conv5, self.conv5_BN)

#         return x

#     def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
#         import torch.nn.functional as F
#         """Forward pass through one layer"""
#         x = F.pad(x, padding)

#         # n, c, h, w = x.shape
#         # _, _, up, down = padding
#         # zeros_up   = torch.zeros([n, c, up, w], device=x.device)
#         # zeros_down = torch.zeros([n, c, down, w], device=x.device)

#         # x = torch.cat([zeros_up, x, zeros_down], dim=2)

#         x = conv(x)
#         x = F.relu(x)
#         x = batch_norm(x)
#         return F.max_pool2d(x, (2, 1), (2, 1))


if __name__ == "__main__":

    full_or_tiny = "tiny"

    model_torch = Crepe_torch(full_or_tiny)
    device = "cuda:0"
    file = f"~/Desktop/PaddleMIX/paddlemix/models/vits-svc/crepe/assets/{full_or_tiny}.pth"
    file = os.path.expanduser(file)
    model_torch.load_state_dict(
            torch.load(file, map_location=device)
            )

    model_paddle = Crepe(full_or_tiny)
    model_paddle_state = model_paddle.state_dict()
    model_torch_state = model_torch.state_dict()

    for key_torch, value_torch in model_torch_state.items():

        if "classifier.weight" == key_torch:
            model_paddle_state[key_torch] = paddle.to_tensor(value_torch.numpy()).T
            continue

        if 'num_batches_tracked' in key_torch:
            # Paddle 的BN中没有这个
            continue

        if key_torch in model_paddle_state:
            assert model_paddle_state[key_torch].shape == list(value_torch.shape)
            model_paddle_state[key_torch] = paddle.to_tensor(value_torch.numpy())
            continue

        if "BN" in key_torch and "running_mean" in key_torch:
            key_paddle = key_torch.replace("running_mean", "_mean")
            assert model_paddle_state[key_paddle].shape == list(value_torch.shape)
            model_paddle_state[key_paddle] = paddle.to_tensor(value_torch.numpy())
            continue

        if "BN" in key_torch and "running_var" in key_torch:
            key_paddle = key_torch.replace("running_var", "_variance")
            assert model_paddle_state[key_paddle].shape == list(value_torch.shape)
            model_paddle_state[key_paddle] = paddle.to_tensor(value_torch.numpy())
            continue

        print(f"{key_torch} 没有经过转换")


    model_paddle.load_dict(model_paddle_state)

    np.random.seed(1942)
    x_numpy = np.random.randn(512, 1024).astype("float32")
    x_paddle = paddle.to_tensor(x_numpy)
    x_torch = torch.from_numpy(x_numpy).cuda()
    model_torch = model_torch.cuda()
    model_paddle.eval()
    model_torch.eval()


    y_torch = model_torch(x_torch)

    y_paddle = model_paddle(x_paddle)

    print( y_torch.mean().item(),  y_torch.std().item() )
    print( y_paddle.mean().item(), y_paddle.std().item() )

    paddle.save(model_paddle.state_dict(), f"crepe/assets/{full_or_tiny}.pdparam")