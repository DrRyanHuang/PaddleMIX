import sys
sys.path.append(
    '/home/zkyd/Desktop/PaddleMIX/paddlemix/models/vits-svc/paddle_project/utils'
    )
import paddle_aux
import paddle
import paddle
import numpy as np


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
            out_0 = paddle.create_parameter(shape=(paddle.randn(shape=[
                n_heads_rel, window_size * 2 + 1, self.k_channels]) *
                rel_stddev).shape, dtype=(paddle.randn(shape=[n_heads_rel, 
                window_size * 2 + 1, self.k_channels]) * rel_stddev).numpy(
                ).dtype, default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.
                k_channels]) * rel_stddev))
            out_0.stop_gradient = not True
            self.emb_rel_k = out_0
            out_1 = paddle.create_parameter(shape=(paddle.randn(shape=[
                n_heads_rel, window_size * 2 + 1, self.k_channels]) *
                rel_stddev).shape, dtype=(paddle.randn(shape=[n_heads_rel, 
                window_size * 2 + 1, self.k_channels]) * rel_stddev).numpy(
                ).dtype, default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[n_heads_rel, window_size * 2 + 1, self.
                k_channels]) * rel_stddev))
            out_1.stop_gradient = not True
            self.emb_rel_v = out_1
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
        x = query.view(b, self.n_heads, self.k_channels, t_t)
        perm_0 = list(range(x.ndim))
        perm_0[2] = 3
        perm_0[3] = 2
        query = x.transpose(perm=perm_0)
        x = key.view(b, self.n_heads, self.k_channels, t_s)
        perm_1 = list(range(x.ndim))
        perm_1[2] = 3
        perm_1[3] = 2
        key = x.transpose(perm=perm_1)
        x = value.view(b, self.n_heads, self.k_channels, t_s)
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
            padded_relative_embeddings = F.pad(relative_embeddings,
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
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, 
            length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:,
            :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = tuple(x.shape)
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length -
            1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]])
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
