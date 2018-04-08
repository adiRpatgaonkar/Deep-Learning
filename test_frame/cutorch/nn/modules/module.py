""" NN module class """

from __future__ import print_function

from collections import OrderedDict

__dlevel__ = 0


class Module(object):
    """ Base class for all nn modules """
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._backward_hooks = OrderedDict()

    def forward(self, *inputs):
        """
        Should be overridden by every subclass module
        """
        raise NotImplementedError

    def backward(self, *inputs):
        """
        Should be overridden by every subclass module
        """
        return self.backward(*inputs)

    def __call__(self, *inputs):
        result = self.forward(*inputs)
        return result

    def update_parameters(self):
        return self.update_parameters()

    def _add_module(self, idx, module):
        self._modules[idx] = module
        module.idx = idx
    
    def _add_parameters(self, idx, module):
        self._parameters[idx] = module._parameters

    def _add_forward_hooks(self, module):
        if module not in self._forward_hooks.values():
            self._forward_hooks[str(len(self._forward_hooks.keys()))] = module

    def _add_backward_hooks(self):
        """
        Not required by the Sequential Container
        """
        raise NotImplementedError

    def see_modules(self):
        print("{")
        for idx, module in self._modules.items():
            print(" {}. {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            print("")
        print("}")
