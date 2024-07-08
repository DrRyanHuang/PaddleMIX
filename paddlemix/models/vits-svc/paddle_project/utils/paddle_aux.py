
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle

def view(self, *args, **kwargs):
    if args:
        if len(args)==1:
            if isinstance(args[0], (tuple, list)):
                return paddle.reshape(self, args[0]) # To change reshape => view
            elif isinstance(args[0], str):
                return paddle.view(self, args[0])
            else:
                return paddle.reshape(self, list(args)) # To change reshape => view
        else:
            return paddle.reshape(self, list(args)) # To change reshape => view
    elif kwargs:
        key = [k for k in kwargs.keys()]
        if 'dtype' in kwargs:
            return paddle.view(self, shape_or_dtype = kwargs[key[0]])
        else:
            return paddle.reshape(self, shape = kwargs[key[0]]) # To change reshape => view

setattr(paddle.Tensor, 'view', view)
