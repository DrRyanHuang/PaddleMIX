
import paddle
# x = paddle.arange(323*2434*2, dtype=paddle.float32).reshape([323, 2434, 2])
y = paddle.randn([1, 513, 3474], dtype="float32") + 1j * paddle.randn([1, 513, 3474], dtype="float32")
# y = paddle.as_complex(x)

# z_as_real = paddle.as_real(y)
z_as_real = y.as_real()

z_stack = paddle.stack(
    [y.real(), y.imag()], axis=-1
)

print(
    (z_as_real == z_stack).all()
)
