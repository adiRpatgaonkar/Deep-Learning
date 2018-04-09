"""Container class to hold nn model layers"""

from __future__ import print_function

import cutorch
from .module import Module
from collections import OrderedDict

__dlevel__ = 0


class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.gradients = OrderedDict()
        for idx, module in enumerate(modules):
            self._add_module(str(idx), module)
            self._add_parameters(str(idx), module)
            self._add_forward_hooks(module)
        self._backward_hooks = OrderedDict(
                                reversed(self._forward_hooks.items())
                                )

    def __getitem__(self, x):
        item, idx = x
        if item == 'module':
            return self._modules.items()[idx]
        if item == 'parameters':
            return self._parameters.items()[idx]

    def forward(self, inputs):
        # print("Input:{}".format(inputs))
        for module in self._modules.values():
            inputs = module(inputs)
        self.output = inputs
        return self

    def backward(self, targets):
        gradients = 0
        for module in self._backward_hooks.values():
            # Pass targets to classifier
            if type(module).__name__ == 'Softmax':
                gradients = module.backward(targets)
            else:
                gradients = module.backward(gradients)
            self.gradients[module] = gradients
        # self.update_parameters()

    def parameters(self):
        # Parameter modules of a container
        return self._parameters

    def update_parameters(self, lr):
        for module in self._modules.values():
            for param in module._parameters:
                param.data = param.data + (-lr * param.gradient)
