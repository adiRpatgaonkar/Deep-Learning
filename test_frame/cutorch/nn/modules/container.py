"""Container class to hold nn model layers"""

from .module import Module

__dlevel__ = 0


class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        for idx, module in enumerate(modules):
            self._add_module(str(idx), module)
            self._add_parameters(str(idx), module)

    def __getitem__(self, x):
        item, idx = x
        if item == 'module':
            return self._modules.items()[idx]
        if item == 'parameters':
            return self._parameters.items()[idx]

    def forward(self, inputs):
        print("Input:{}".format(inputs))
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs

    def parameters(self):
        # Parameter modules of a container
        return self._parameters
