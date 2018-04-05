""" NN module class """

from __future__ import print_function

from collections import OrderedDict


class Module(object):
    """ Base class for all nn modules """
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self._forward_hooks = OrderedDict()

    def __call__(self, inputs):
    	return self.forward(inputs)
    	


    def _add_module(self, idx, module):
        self._modules[idx] = module
    
    def _add_parameters(self, idx, module):
        self._parameters[idx] = module._parameters

    def _add_forward_hooks(self):
    	for value in self.__dict__.values():
    		if type(value).__name__ == 'Sequential':
    			self._forward_hooks[value] = value._modules

    def _forward(self, *inputs):
        """
        Should be overridden by every subclass module
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
