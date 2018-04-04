""" NN module class """

from __future__ import print_function

from collections import OrderedDict

global DEBUG
DEBUG = False

class Module(object):
    """ Base class for all nn modules """
    def __init__(self):
        self.x = 0
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

    def forward(self, *input):
        """
        Should be overriden by every module
        """
        t = 0

    def add_module(self, name, module):
    	self._modules[name] = module
    	if 'weight' in module.__dict__:
    		self._parameters[module] = module.parameters
    	
    def parameters(self):
    	return self._parameters

    def see_modules(self):
    	print("{")
    	for idx, module in self._modules.items():
    		print(" {}. {} ".format(idx, type(module).__name__), end="")
    		if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
    			print("({}x{})".format(module.in_features, module.out_features), end="")
    		print("")
    	print("}")