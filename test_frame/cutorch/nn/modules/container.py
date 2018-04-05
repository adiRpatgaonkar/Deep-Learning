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
        for module in self._modules.values():
            if __debug__:
                if __dlevel__ == 2:
                    print(inputs)
                    print(type(module).__name__)
            inputs = module(inputs)
        return inputs

    def parameters(self):
        return self._parameters
