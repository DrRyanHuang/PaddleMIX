import math
import numpy as np
import torch
import paddle
# from torch import nn
# from torch.nn import functional as F

# from vits import commons
# from vits.modules import LayerNorm


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class LayerNorm_torch(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

        # import torch.nn.init as init

        # 使用均匀分布随机初始化 gamma 和 beta
        # self.gamma = torch.nn.Parameter(torch.rand(channels))
        # self.beta = torch.nn.Parameter(torch.rand(channels))


    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)



class LayerNorm(paddle.nn.Layer):

    def __init__(self, channels, eps=1e-05):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = self.create_parameter(shape=[channels], 
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(
            shape=channels)))

        self.beta = self.create_parameter(shape=[channels],
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=channels)))

    def forward(self, x):

        perm_0 = list(range(x.ndim))
        perm_0[1] = -1
        perm_0[-1] = 1
        x = x.transpose(perm=perm_0)
        x = paddle.nn.functional.layer_norm(x=x, 
                                            normalized_shape=(self.channels,), 
                                            weight=self.gamma, 
                                            bias=self.beta, 
                                            epsilon=self.eps)

        perm_1 = list(range(x.ndim))
        perm_1[1] = -1
        perm_1[-1] = 1
        return x.transpose(perm=perm_1)


def LayerNorm_torch2paddle(torch_model, paddle_model):
    paddle_model.gamma.set_value(
        paddle.to_tensor(
            torch_model.gamma.detach().cpu().numpy()
        )
    )

    paddle_model.beta.set_value(
        paddle.to_tensor(
            torch_model.beta.detach().cpu().numpy()
        )
    )

if __name__ == "__main__":

    channal = 8

    torch_model = LayerNorm_torch(channal).cuda()
    paddle_model = LayerNorm(channal)

    x = np.random.rand(32, 8, channal).astype("float32")
    x_tc = torch.from_numpy(x).cuda()
    x_pd = paddle.to_tensor(x)

    LayerNorm_torch2paddle(torch_model, paddle_model)

    y_tc = torch_model(x_tc).detach().cpu().numpy()
    y_pd = paddle_model(x_pd).detach().cpu().numpy()

     
    print(
        abs(y_tc - y_pd).max().item()
    )






















class Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         hidden_channels,
#         filter_channels,
#         n_heads,
#         n_layers,
#         kernel_size=1,
#         p_dropout=0.0,
#         proximal_bias=False,
#         proximal_init=True,
#         **kwargs
#     ):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.filter_channels = filter_channels
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.kernel_size = kernel_size
#         self.p_dropout = p_dropout
#         self.proximal_bias = proximal_bias
#         self.proximal_init = proximal_init

#         self.drop = nn.Dropout(p_dropout)
#         self.self_attn_layers = nn.ModuleList()
#         self.norm_layers_0 = nn.ModuleList()
#         self.encdec_attn_layers = nn.ModuleList()
#         self.norm_layers_1 = nn.ModuleList()
#         self.ffn_layers = nn.ModuleList()
#         self.norm_layers_2 = nn.ModuleList()
#         for i in range(self.n_layers):
#             self.self_attn_layers.append(
#                 MultiHeadAttention(
#                     hidden_channels,
#                     hidden_channels,
#                     n_heads,
#                     p_dropout=p_dropout,
#                     proximal_bias=proximal_bias,
#                     proximal_init=proximal_init,
#                 )
#             )
#             self.norm_layers_0.append(LayerNorm(hidden_channels))
#             self.encdec_attn_layers.append(
#                 MultiHeadAttention(
#                     hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
#                 )
#             )
#             self.norm_layers_1.append(LayerNorm(hidden_channels))
#             self.ffn_layers.append(
#                 FFN(
#                     hidden_channels,
#                     hidden_channels,
#                     filter_channels,
#                     kernel_size,
#                     p_dropout=p_dropout,
#                     causal=True,
#                 )
#             )
#             self.norm_layers_2.append(LayerNorm(hidden_channels))

#     def forward(self, x, x_mask, h, h_mask):
#         """
#         x: decoder input
#         h: encoder output
#         """
#         self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
#             device=x.device, dtype=x.dtype
#         )
#         encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
#         x = x * x_mask
#         for i in range(self.n_layers):
#             y = self.self_attn_layers[i](x, x, self_attn_mask)
#             y = self.drop(y)
#             x = self.norm_layers_0[i](x + y)

#             y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
#             y = self.drop(y)
#             x = self.norm_layers_1[i](x + y)

#             y = self.ffn_layers[i](x, x_mask)
#             y = self.drop(y)
#             x = self.norm_layers_2[i](x + y)
#         x = x * x_mask
#         return x


class MultiHeadAttention_torch(torch.nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)





class MultiHeadAttention(paddle.nn.Layer):

    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0,
        window_size=None, heads_share=True, block_length=None,
        proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        self.conv_k = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        self.conv_v = paddle.nn.Conv1D(in_channels=channels, out_channels=
            channels, kernel_size=1)
        self.conv_o = paddle.nn.Conv1D(in_channels=channels, out_channels=
            out_channels, kernel_size=1)
        self.drop = paddle.nn.Dropout(p=p_dropout)

        if window_size is not None:

            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5

            self.emb_rel_k = paddle.create_parameter(
                shape=[n_heads_rel, window_size * 2 + 1, self.k_channels], 
                dtype="float32",
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.k_channels]) * rel_stddev))

            self.emb_rel_v = paddle.create_parameter(
                shape=[n_heads_rel, window_size * 2 + 1, self.k_channels], 
                dtype="float32",
                default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.k_channels]) * rel_stddev))

        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_q.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_k.weight)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.conv_v.weight)

        if proximal_init:
            with paddle.no_grad():
                paddle.assign(self.conv_q.weight, output=self.conv_k.weight)
                paddle.assign(self.conv_q.bias, output=self.conv_k.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):

        b, d, t_s, t_t = *tuple(key.shape), query.shape[2]
        x = query.view([b, self.n_heads, self.k_channels, t_t])

        perm_0 = list(range(x.ndim))
        perm_0[2] = 3
        perm_0[3] = 2
        query = x.transpose(perm=perm_0)
        x = key.view([b, self.n_heads, self.k_channels, t_s])

        perm_1 = list(range(x.ndim))
        perm_1[2] = 3
        perm_1[3] = 2
        key = x.transpose(perm=perm_1)
        x = value.view([b, self.n_heads, self.k_channels, t_s])

        perm_2 = list(range(x.ndim))
        perm_2[2] = 3
        perm_2[3] = 2
        value = x.transpose(perm=perm_2)

        x = key
        perm_3 = list(range(x.ndim))
        perm_3[-2] = -1
        perm_3[-1] = -2
        scores = paddle.matmul(x=query / math.sqrt(self.k_channels), y=x.
            transpose(perm=perm_3))

        if self.window_size is not None:
            assert t_s == t_t, 'Relative attention is only available for self-attention.'
            key_relative_embeddings = self._get_relative_embeddings(self.
                emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(
                self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(
                rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias is only available for self-attention.'
            scores = scores + self._attention_bias_proximal(t_s).to(device=
                scores.place, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask=mask == 0, value=-10000.0)
            if self.block_length is not None:
                assert t_s == t_t, 'Local attention is only available for self-attention.'
                block_mask = paddle.ones_like(x=scores).triu(-self.block_length
                    ).tril(self.block_length)
                scores = scores.masked_fill(mask=block_mask == 0, value=-
                    10000.0)
        p_attn = paddle.nn.functional.softmax(F, axis=-1)
        p_attn = self.drop(p_attn)
        output = paddle.matmul(x=p_attn, y=value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.
                emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)

        x = output
        perm_4 = list(range(x.ndim))
        perm_4[2] = 3
        perm_4[3] = 2
        output = x.transpose(perm=perm_4).view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = paddle.matmul(x=x, y=y.unsqueeze(axis=0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        x = y.unsqueeze(axis=0)
        perm_5 = list(range(x.ndim))
        perm_5[-2] = -1
        perm_5[-1] = -2
        ret = paddle.matmul(x=x, y=x.transpose(perm=perm_5))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = paddle.nn.functional.pad(relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
            slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = tuple(x.shape)
        x = paddle.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        # x_flat = paddle.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, 
        #     length - 1]]))
        x_flat = paddle.nn.functional.pad(x_flat, [0, length - 1], data_format="NCL")
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:,
            :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = tuple(x.shape)
        x = paddle.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length -
            1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = paddle.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]])
            )
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = paddle.arange(dtype='float32', end=length)
        diff = paddle.unsqueeze(x=r, axis=0) - paddle.unsqueeze(x=r, axis=1)
        return paddle.unsqueeze(x=paddle.unsqueeze(x=-paddle.log1p(x=paddle
            .abs(x=diff)), axis=0), axis=0)


def MultiHeadAttention_torch2paddle(torch_model, paddle_model):

    paddle_model.conv_q.weight.set_value(
        paddle.to_tensor(
            torch_model.conv_q.weight.detach().cpu().numpy()
        )
    )
    paddle_model.conv_q.bias.set_value(
        paddle.to_tensor(
            torch_model.conv_q.bias.detach().cpu().numpy()
        )
    )

    paddle_model.conv_k.weight.set_value(
        paddle.to_tensor(
            torch_model.conv_k.weight.detach().cpu().numpy()
        )
    )
    paddle_model.conv_k.bias.set_value(
        paddle.to_tensor(
            torch_model.conv_k.bias.detach().cpu().numpy()
        )
    )

    paddle_model.conv_v.weight.set_value(
        paddle.to_tensor(
            torch_model.conv_v.weight.detach().cpu().numpy()
        )
    )
    paddle_model.conv_v.bias.set_value(
        paddle.to_tensor(
            torch_model.conv_v.bias.detach().cpu().numpy()
        )
    )

    paddle_model.conv_o.weight.set_value(
        paddle.to_tensor(
            torch_model.conv_o.weight.detach().cpu().numpy()
        )
    )
    paddle_model.conv_o.bias.set_value(
        paddle.to_tensor(
            torch_model.conv_o.bias.detach().cpu().numpy()
        )
    )

if __name__ == "__main__":

    channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init = \
        (192, 192, 2, 0.1, 4, True, None, False, False)

    torch_model = MultiHeadAttention_torch(
        channels, out_channels, n_heads, p_dropout, window_size, heads_share, 
        block_length, proximal_bias, proximal_init
    ).cuda()

    paddle_model = MultiHeadAttention(
        channels, out_channels, n_heads, p_dropout, window_size, heads_share, 
        block_length, proximal_bias, proximal_init
    )

    MultiHeadAttention_torch2paddle(torch_model, paddle_model)

    x = np.random.rand(8, 192, 400).astype("float32")
    c = np.random.rand(8, 192, 400).astype("float32")
    attn_mask = np.random.rand(8, 1, 400, 400).astype("float32")

    x_tc = torch.from_numpy(x).cuda()
    c_tc = torch.from_numpy(c).cuda()
    attn_mask_tc = torch.from_numpy(attn_mask).cuda()

    x_pd = paddle.to_tensor(x)
    c_pd = paddle.to_tensor(c)
    attn_mask_pd = paddle.to_tensor(attn_mask)

    # y_tc = torch_model(x_tc, c_tc, attn_mask_tc).detach().cpu().numpy()
    y_pd = paddle_model(x_pd, c_pd, attn_mask_pd).detach().cpu().numpy()

    print(
        abs(y_tc - y_pd).max().item()
    )










class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1D(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1D(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = paddle.nn.functional.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = paddle.nn.functional.pad(x, convert_pad_shape(padding))
        return x
