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
        self.input = None # CLEAN
        self.data = 0 # CLEAN
        # Parameters' creation
        self._parameters = []
        self.weight = Parameter(weight=beta*torch.randn(self.in_features, self.out_features),
                                require_gradient=True)
        if bias is True:
            self.bias = Parameter(bias=torch.Tensor(1, out_features).fill_(0),
                                  require_gradient=True)
        # Gradients' creation
        self.grad = OrderedDict() # CLEAN
        if self.weight.require_gradient:
            self.grad['weight'] = 0
        if bias and self.bias.require_gradient:
            self.grad['bias'] = 0
        self.grad['output'] = 0
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

    def backward(self, gradients):
        # print(self.input.t(), gradients['output'])
        if gradients['output'].dim() == 1:
            gradients['output'] = gradients['output'].unsqueeze(0)

        if self.weight.require_gradient:
            self.grad['weight'] = f.gradient_weight(self.input, gradients['output'])
            self.weight.gradient = self.grad['weight']

        if self.bias.require_gradient:
            self.grad['bias'] = f.gradient_bias(gradients['output'])
            self.bias.gradient = self.grad['bias']
        if self.idx == '0':
            # No gradients required for input layer (idx == 0)
            self.grad['output'] = torch.Tensor(self.input.size())
        else:
            self.grad['output'] = f.gradient_linear(self.weight.data, gradients['output'])
        return self.grad
