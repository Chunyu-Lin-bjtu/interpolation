from torch.nn.modules.module import Module

from ..functions.resample2d import Resample2dFunction
from ..functions.resample2d import ForwardResample2dFunction


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)

class ForwardResample2d(Module):

    def __init__(self, kernel_size=1):
        super(ForwardResample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return ForwardResample2dFunction.apply(input1_c, input2, self.kernel_size)
