"""Container class to hold nn model layers"""

from .module import Module

global DEBUG
DEBUG = False

class Sequential(Module):

	def __init__(self, *modules):
		super(Sequential, self).__init__()

		for idx, module in enumerate(modules):
			self.add_module(str(idx), module)

	def forward(self, inputs):
		for module in self._modules.values():
			if DEBUG:
				print(inputs)
				print(type(module).__name__)
			inputs = module.forward(inputs)
		return inputs