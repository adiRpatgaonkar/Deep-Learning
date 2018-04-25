from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as f

class Conv2D(Module):
    """2D Conv layer class"""

    def __init__(self, kernels, kernel_size=3, pad=0, stride=1):
        super(Conv2D, self).__init__()
        self.kernel_size = kernel_size
        self.kernels, self.padding, self.stride = kernels, pad, stride
        self.feature_map_volume = torch.zeros(0, 0)

    def forward(self, in_features):
        """ Convolution op (Auto-correlation i.e. no flipping of kernels)"""
        self.create_output_vol()
        if torch.is_tensor(in_features):
            in_features = in_features.numpy()
        # F.pad
        # in_features = np.pad(in_features,
        #                      mode='constant', constant_values=0,
        #                      pad_width=((0, 0), (0, 0), 
        #                      (self.padding, self.padding), 
        #                      (self.padding, self.padding)))
        in_features = torch.from_numpy(in_features) # TODO: Enable cuda compatibilty
        self.input = in_features
        fh = fw = self.kernels.size(2)
        # F.im2col
        # convolve
        return self
    
    def create_output_vol(self):
        self.depth, self.height, self.width = self.input.size()
        self.kernels = 0.01 * torch.randn(self.kernels, self.depth, self.kernel_size, self.kernel_size)
        self.biases = torch.ones(self.kernels, 1, 1, 1)  # print(self.biases)
        self.output_dim = [0, 0, 0]
        self.output_dim[0] = self.kernels
        self.output_dim[1] = ((self.width - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        print(self.output_dim, self.feature_map_volume.size(), self.kernels.size())