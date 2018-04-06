"""
Activations class
1. ReLU
"""

from __future__ import print_function

from .module import Module
from .. import functionals as f

__dlevel__ = 0


class ReLU(Module):
    """ReLU Activation layer class"""

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, in_features):
        self.activ_tensor = f.relu(in_features) 
        if __debug__:
            if __dlevel__ == 4:
                print(self.activ_tensor)
        return self.activ_tensor

    # @staticmethod
    # def backward_relu(inputs, grad_outputs):
    #     grad_outputs[inputs <= 0] = 0
    #     return grad_outputs

class Softmax(Module):
    """ Softmax class """

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, in_tensor):
        self.softmaxed_tensor = f.softmax(in_tensor)
        return self.softmaxed_tensor

    def predict(softmaxed_in):
        self.value, self.index = torch.max(self.softmaxed_tensor, 1)
        return value, index