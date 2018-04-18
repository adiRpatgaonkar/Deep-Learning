"""Container class to hold nn model layers"""

from __future__ import print_function

from collections import OrderedDict as OD

import cutorch
from .module import Module

__dlevel__ = 0


class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        for idx, module in enumerate(modules):
            self._add_module(str(idx), module)
            self._add_parameters(str(idx), module)

    def forward(self, inputs):
        # print("Input:{}".format(inputs))
        for module in self.modules.values():
            inputs = module(inputs)
        self.data = inputs.data
        return inputs