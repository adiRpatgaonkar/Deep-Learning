from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F

class Conv2d(Module):
    """2D Conv layer class"""

    def __init__(self, channels, kernels, kernel_size=3, pad=0, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.kernels, self.padding, self.stride = kernels, pad, stride
        # Setup weights(kernels) and biases
        self.weight = 0.01 * torch.randn(self.kernels, channels, self.kernel_size, self.kernel_size)
        if bias:
            self.bias = torch.ones(self.kernels, 1, 1, 1)  # print(self.biases)
        # Layer construct check
        if pad < 0:
            raise ValueError("Invalid padding value. Should be >= 0")
        if stride <= 0:
            raise ValueError("Invalid padding value. Should be > 0")

    def create_output_vol(self):
        """ Create output volume """
        # Check & setup input tensor dimensions
        # Input should be a batch or 3D tensor
        if self.input.dim() not in (3, 4):
            raise ValueError("Input tensor should be 3D or 4D")
        elif self.input.dim() == 3:
            self.input = torch.unsqueeze(self.input, 0)
        self.height, self.width = self.input.size()[2:]
        self.output_dim = [0, 0, 0] # For a single image
        self.output_dim[0] = self.kernels
        self.output_dim[1] = ((self.width - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.output_dim[2] = ((self.height - self.kernel_size + 2 * self.padding) / self.stride + 1)

    def prepare_input(self):
        """ Prepare in features """
        # 1. Padding: self.input should be 4D
        if self.padding > 0:
            self.input = F.pad_image(self.input, self.padding)
            if type(self.input) is np.ndarray: # If numpy array, convert to tensor before conv op.
                self.input = torch.from_numpy(self.input)
        # 2. im2col operation (One image @ a time.)
        batch_ims = torch.Tensor()
        for image in self.input:
            batch_ims = torch.cat((batch_ims, F.im2col(image, self.kernel_size, self.stride, task="conv").unsqueeze_(0)), 0)
        return batch_ims

    def forward(self, in_features):
        """ Convolution op (Auto-correlation i.e. no kernel flipping) """
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        print("Input to conv layer:", self.input.size())
        self.create_output_vol()
        self.input = self.prepare_input() # im2col'ed input
        # print("Post im2col:", self.input.size())
        self.data = F.conv_2d(self.input, self.weight.view(self.weight.size(0), -1), 
                              self.bias.view(self.bias.size(0), -1))
        # Reshape to the o/p feature volume 
        self.data = self.data.view(self.data.size(0), self.output_dim[0], 
                                   self.output_dim[1], self.output_dim[2])
        # print("Reshaped:", self.data.size())
        return self

    def backward(self):
        pass