import paddle
import torch
# import torch.nn as nn
# import torch.nn.functional as F


class ScaleDiscriminator_torch(torch.nn.Module):
    def __init__(self):
        super(ScaleDiscriminator_torch, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = torch.nn.utils.weight_norm(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return [(fmap, x)]




class ScaleDiscriminator(paddle.nn.Layer):

    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = paddle.nn.LayerList(sublayers=[
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=7)), 
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=16, out_channels=64, kernel_size=41, stride=4, groups=4, padding=20)), 
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=64,out_channels=256, kernel_size=41, stride=4, groups=16, padding=20)), 
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=256, out_channels=1024, kernel_size=41, stride=4,groups=64, padding=20)), 
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=1024, out_channels=1024,kernel_size=41, stride=4, groups=256, padding=20)), 
            paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=1024,out_channels=1024, kernel_size=5, stride=1, padding=2))
            ])
        self.conv_post = paddle.nn.utils.weight_norm(layer=paddle.nn.Conv1D(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return [(fmap, x)]




def ScaleDiscriminator_torch2paddle(torch_model, paddle_model):

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
        paddle.to_tensor( torch_model.conv_post.weight_g.detach().cpu().numpy()[0,0] )
    )
    paddle_model.conv_post.bias.set_value(
        paddle.to_tensor( torch_model.conv_post.bias.detach().cpu().numpy() )
    )


if __name__ == "__main__":

    torch_model = ScaleDiscriminator_torch().cuda()
    paddle_model = ScaleDiscriminator()

    import numpy as np
    x_np = np.random.rand(8, 1, 8000).astype("float32")
    x_tc = torch.from_numpy(x_np).cuda()
    x_pd = paddle.to_tensor(x_np)

    ScaleDiscriminator_torch2paddle(torch_model, paddle_model)

    _, y_tc = torch_model(x_tc)[0]
    _, y_pd = paddle_model(x_pd)[0]

    y_tc = y_tc.detach().cpu().numpy()
    y_pd = y_pd.detach().cpu().numpy()

    print(
        "ScaleDiscriminator",
        abs(y_tc - y_pd).max().item()
    )

