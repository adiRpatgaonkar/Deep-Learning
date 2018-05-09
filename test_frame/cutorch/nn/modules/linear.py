""" Linear layer class """

from __future__ import print_function

from collections import OrderedDict

import torch

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as f


class Linear(Module):
    """Linear Layer class"""

    def __init__(self, in_features, out_features, bias=True, beta=0.01):
        super(Linear, self).__init__()
        # self.parent = None # No use as of now.
        self.idx = -1
        self.in_features = in_features
        self.out_features = out_features
        self.input = None  # CLEAN
        self.data = 0  # CLEAN
        # Parameters' creation
        self._parameters = []
        self.weight = Parameter(weight=beta*torch.randn(self.in_features, self.out_features),
                                require_gradient=True)
        if bias is True:
            self.bias = Parameter(bias=torch.Tensor(1, out_features).fill_(0),
                                  require_gradient=True)
        # Grad input
        self.grad_in = None
        # Finish param setup
        self.init_param_setup()

    def parameters(self):
        return self._parameters

    def init_param_setup(self):
        self._parameters.append(self.weight)
        if 'bias' in self.__dict__:
            self._parameters.append(self.bias)

    def forward(self, in_features):
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        self.data = f.linear(self.input, self.weight.data, self.bias.data)
        return self

    def backward(self, grad_out):
        # gradients['in'] are actually output gradients
        # grad['in'] are actual input gradients
        # print(self.input.t(), gradients['in'])
        if grad_out.dim() == 1:
            grad_out = grad_out.unsqueeze(0)

        if self.weight.require_gradient:
            self.weight.grad = f.gradient_weight(self.input, grad_out)

        if self.bias.require_gradient:
            self.bias.grad = f.gradient_bias(grad_out)
        if self.idx == '0':
            # No gradients required for input layer (idx == 0)
            self.grad_in = torch.Tensor(self.input.size())
        else:
            self.grad_in = f.gradient_linear(self.weight.data, grad_out)
        return self.grad_in

