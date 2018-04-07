"""
Activations class
1. ReLU
"""

from __future__ import print_function

import torch

from .module import Module
from .. import functionals as f

__dlevel__ = 0


class ReLU(Module):
    """ReLU Activation layer class"""

    def backward(self, *inputs):
        pass

    def __init__(self):
        super(ReLU, self).__init__()
        self.output = 0
        self.grad = 0

    def forward(self, in_features):
        self.output = f.relu(in_features)
        if __debug__:
            if __dlevel__ == 4:
                print(self.output)
        return self.output

    # @staticmethod
    # def backward_relu(inputs, grad_outputs):
    #     grad_outputs[inputs <= 0] = 0
    #     return grad_outputs


class Softmax(Module):
    """ Softmax class """

    def backward(self, *inputs):
        pass

    def __init__(self):
        super(Softmax, self).__init__()
        self.output = 0
        self.grad = 0

    def forward(self, in_tensor):
        self.output = f.softmax(in_tensor)
        return self.output

    def predict(self, softmaxed_in):
        self.value, self.index = torch.max(softmaxed_in, 1)
        return [self.value, self.index]
