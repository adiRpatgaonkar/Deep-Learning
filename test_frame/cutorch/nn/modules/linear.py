""" Linear layer class """

from __future__ import print_function

import torch

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as f


class Linear(Module):
    """Linear Layer class"""

    def __init__(self, in_features, out_features, bias=True):
        # print('Linear layer created')
        # allocate size for the state variables appropriately
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parameters = []
        self.weight = Parameter(weight=torch.randn(self.in_features,
                                                   self.out_features))
        self.parameters.append(self.weight)
        if bias:
            self.bias = Parameter(bias=torch.Tensor(1,
                                                    out_features).fill_(0))
            self.parameters.append(self.bias)
        if __debug__:
            print(self.weight.tag, self.weight.data)
            print(self.bias.tag, self.bias.data)

    def forward(self, in_features):
        if __debug__:
            print(type(self).__name__)
            print(f.linear(in_features, self.weight.data, self.bias.data))
        return f.linear(in_features, self.weight.data, self.bias.data)
