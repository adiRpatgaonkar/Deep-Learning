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
        raise NotImplementedError

    def __call__(self, *inputs):
        for mem in self.__dict__.values():
            if 'Sequential' in self.__dict__:
                self._add_forward_hooks(mem)
        result = self.forward(*inputs)
        self._add_backward_hooks()
        return result

    def _add_module(self, idx, module):
        self._modules[idx] = module
    
    def _add_parameters(self, idx, module):
        self._parameters[idx] = module._parameters

    def _add_forward_hooks(self, container):
        if not container in self._forward_hooks.values():
            self._forward_hooks[str(len(self._forward_hooks.keys()))] = container

    def _add_backward_hooks(self):
        pass

    def see_modules(self):
        print("{")
        for idx, module in self._modules.items():
            print(" {}. {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            print("")
        print("}")
