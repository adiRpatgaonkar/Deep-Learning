"""
Batchnorm layer class
1. BatchNorm2d 
"""

from __future__ import print_function

from collections import OrderedDict

import torch

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F


class BatchNorm2d(Module):
    """ Batch Norm 2D module. *Under construction* """
    def __init__(self, num_features, beta=None, gamma=None, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        self.idx = -1
        self.channels = num_features
        self.mean = 0  # running
        self.variance = 0  # running
        self.epsilon = eps
        self.input = None  # TODO:CLEAN
        # Intermediate terms used for backward
        self.cache = None  # TODO:CLEAN
        self.data = 0  # TODO:CLEAN
        # Parameters' creation
        self._parameters = []
        if beta is None:
            self.beta = Parameter(beta=torch.zeros(self.channels), 
                                  require_gradient=True)
        if gamma is None:
            self.gamma = Parameter(gamma=torch.ones(self.channels),
                                   require_gradient=True)
        # Gradients' creation
        self.grad = OrderedDict()  # TODO:CLEAN
        if self.beta.require_gradient:
            self.grad['beta'] = 0
        if self.gamma.require_gradient:
            self.grad['gamma'] = 0
        self.grad['input'] = 0
        # Finish param setup
        self.init_param_setup()

    def parameters(self):
        return self._parameters

    def init_param_setup(self):
        self._parameters.append(self.beta)
        self._parameters.append(self.gamma)

    def forward(self, in_features):
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        assert self.input.size(1) == self.channels, ("input channels should be {}".format(self.channels))
        print("Input to B_normed2d layer:", self.input.size())
        self.data, self.mean, self.variance, self.cache = \
        F.batchnorm_2d(self.input, self.beta.data, self.gamma.data, self.epsilon)
        # Clean
        del in_features
        return self

    def backward(self, gradients):
        if self.beta.require_gradient:
            self.grad['beta'] = F.gradient_beta(gradients['input'])
            self.beta.gradient = self.grad['beta']
        if self.gamma.require_gradient:
            self.grad['gamma'] = F.gradient_gamma(gradients['input'])
            self.gamma.gradient = self.grad['gamma']
        self.grad['input'] = F.gradient_bnorm2d(self.gamma.data, self.cache, gradients['input'])
        # Clean
        del gradients 
        return self.grad

