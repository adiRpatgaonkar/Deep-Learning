from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F


class BatchNorm2d(Module):
    """ Batch Norm 2D module. *Under construction* """
    def __init__(self, channels, beta=None, gamma=None, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        self.channels = channels
        self.mean = 0
        self.variance = 0
        self.epsilon = eps
        if beta is None:
            self.beta = torch.zeros(self.channels)
        if gamma is None:
            self.gamma = torch.ones(self.channels)
        self.data = 0

    def forward(self, in_features):
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        assert self.input.size(1) == self.channels, ("input channels should be {}".format(self.channels))
        print("Input to B_normed2d layer:", self.input.size())
        self.mean, self.variance, self.data = \
        F.batchnorm_2d(self.input, self.beta, self.gamma, self.epsilon)
        return self

    def backward(self):
        pass