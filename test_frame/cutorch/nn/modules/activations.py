""" Activations class 
	1. ReLU

"""

from __future__ import print_function

from .module import Module
from .. import functionals as f

global DEBUG
DEBUG = False

class ReLU(Module):
    """ReLU Activation layer class"""

    def __init__(self):
        # Different activation function
        super(ReLU, self).__init__()

    def forward(self, in_features):
    	if DEBUG:
    		print(f.relu(in_features))
    	return f.relu(in_features)

    # @staticmethod
    # def backward_relu(inputs, grad_outputs):
    #     grad_outputs[inputs <= 0] = 0
    #     return grad_outputs