""" Linear layer class """

from __future__ import print_function

from collections import OrderedDict

import torch

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as f


__dlevel__ = 0


class Linear(Module):
    """Linear Layer class"""

    def __init__(self, in_features, out_features, bias=True):
        # print('Linear layer created')
        # allocate size for the state variables appropriately
        super(Linear, self).__init__()
        self.idx = 0
        self.in_features = in_features
        self.out_features = out_features
        self.inputs = torch.Tensor([0.0])
        self.output = 0

        self.weight = Parameter(weight=torch.randn(self.in_features, self.out_features),
                                require_gradient=True)
        if bias is True:
            self.bias = Parameter(bias=torch.Tensor(1, out_features).fill_(0),
                                  require_gradient=True)

        self.grad = OrderedDict()

        if self.weight.require_gradient:
            self.grad['weight'] = 0
        if bias and self.bias.require_gradient:
            self.grad['bias'] = 0
        self.grad['output'] = 0

        self.init_param_setup()

        if __debug__:
            if __dlevel__ == 4:
                print(self.weight.tag, self.weight.data)
                print(self.bias.tag, self.bias.data)
                print(self.output)

    def add2module(self):
        self._parameters = []
        self._parameters.append(self.weight)
        if 'bias' in self.__dict__:
            self._parameters.append(self.bias)

    def init_param_setup(self):
        self.weight.data = f.decay_weight(self.weight.data)
        self.add2module()

    def forward(self, in_features):
        self.inputs = in_features
        self.output = f.linear(in_features, self.weight.data, self.bias.data)
        print(list(self.output))
        if __debug__:
            if __dlevel__ == 4:
                print(type(self).__name__)
                print(self.output.data)
        return self.output

    def backward(self, gradients):
        # print(self.inputs.t(), gradients['output'])
        if gradients['output'].dim() == 1:
            gradients['output'] = gradients['output'].unsqueeze(0)

        if self.weight.require_gradient:
            self.grad['weight'] = f.gradient_weight(self.inputs, gradients['output'])
            self.weight.gradient = self.grad['weight']

        if self.bias.require_gradient:
            self.grad['bias'] = f.gradient_bias(gradients['output'])
            self.bias.gradient = self.grad['bias']
        if self.idx == '0':
            # No gradients required for input layer (idx == 0)
            self.grad['output'] = torch.Tensor(self.inputs.size())
        else:
            self.grad['output'] = f.gradient_linear(gradients['output'], self.weight.data)
        return self.grad
