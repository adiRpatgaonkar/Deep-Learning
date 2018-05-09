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
    def __init__(self, num_features, running=True, beta=None, gamma=None, eps=1e-5):
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
        self.grad_in = None  # CLEAN
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
        #print("Input to B_normed2d layer:", self.input.size())
        if Module.is_train:
            self.data, self.mean, self.variance, self.cache = \
            F.batchnorm_2d(self.input, self.beta.data, self.gamma.data, self.epsilon)
        elif Module.is_eval:
            self.data, _, _, _ = \
            F.batchnorm_2d(self.input, self.beta.data, self.gamma.data, self.epsilon, 
                           r_mean=self.mean, r_var=self.variance)
        return self

    def backward(self, grad_out):
        if self.beta.require_gradient:
            self.beta.grad = F.gradient_beta(grad_out)
        if self.gamma.require_gradient:
            self.gamma.grad = F.gradient_gamma(grad_out)
        self.grad_in = F.gradient_bnorm2d(self.gamma.data, self.cache, grad_out) 
        return self.grad_in

