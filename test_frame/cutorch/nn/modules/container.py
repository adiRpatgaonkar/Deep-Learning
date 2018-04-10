"""Container class to hold nn model layers"""

from __future__ import print_function

import cutorch
from .module import Module
from collections import OrderedDict

__dlevel__ = 0


class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        for idx, module in enumerate(modules):
            self._add_module(str(idx), module)
            self._add_parameters(str(idx), module)
            self._add_forward_hooks(module)
        self._backward_hooks = OrderedDict(
                                reversed(self._forward_hooks.items()))
        self.gradients = OrderedDict()

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
        self.data = inputs
        return self

    def backward(self, targets):
        gradients = targets # Alias for targets of classifier
        for module in self._backward_hooks.values():
            gradients = module.backward(gradients)
            # Store gradients @ current iteration
            # for every module
            #self.gradients[module] = gradients
        # Optimizer does this. So commented out
        # self.update_parameters()

    def eval(self, data):
        """
        :param data: testing data
        :return accuracy: testing accuracy
        """
        pass

    def parameters(self):
        # Parameter modules of a container
        return self._parameters

    def update_parameters(self, lr):
        for module in self._modules.values():
            for param in module._parameters:
                param.data = param.data + (-lr * param.gradient)
