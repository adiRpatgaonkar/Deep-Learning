from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F

class Conv2d(Module):
    """2D Conv layer class"""

    def __init__(self, channels, kernels, kernel_size=3, pad=0, stride=1):
        super(Conv2d, self).__init__()
        self.depth = channels
        self.kernel_size = kernel_size
        self.kernels, self.padding, self.stride = kernels, pad, stride

        # Layer construct check
        if pad < 0:
            raise ValueError("Invalid padding value. Should be >= 0")
        if stride <= 0:
            raise ValueError("Invalid padding value. Should be > 0")

    def create_output_vol(self):
        """ Create output volume """
        self.height, self.width = self.input.size()[2:]
        # Setup weights(kernels).
        self.weight = 0.01 * torch.randn(self.kernels, self.depth, self.kernel_size, self.kernel_size)
        self.bias = torch.ones(self.kernels, 1, 1, 1)  # print(self.biases)
        self.output_dim = [0, 0, 0] # For a single image
        self.output_dim[0] = self.kernels
        self.output_dim[1] = ((self.width - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        # print("Out dims:", self.output_dim, "Feature map:", "Weights:", self.weight.size())

    def prepare_input(self):
        """ Prepare in features """
        if self.input.dim() == 3: # For a single image
            in_features = torch.unsqueeze(self.input, 0)
        elif self.input.dim() == 4:
            in_features = self.input
            
        # 1. Padding
        if self.padding > 0:
            in_features = F.pad_image(in_features, self.padding)
 
        # 2. im2col operation
        if type(in_features) is np.ndarray: # If numpy array, convert to tensor before conv op.
            in_features = torch.from_numpy(in_features) 
        if in_features.dim() == 3: # For single image
            in_features = torch.unsqueeze(in_features, 0)
        batch_ims = torch.Tensor()
        for image in in_features:
            batch_ims = torch.cat((batch_ims, F.im2col(image, self.kernel_size, self.stride, task="conv").unsqueeze_(0)), 0)
        del in_features
        return batch_ims

    def forward(self, in_features):
        """ Convolution op (Auto-correlation i.e. no flipping of kernels)"""
        # Check for tensor input
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        print("Input to conv layer:", self.input.size())
        self.create_output_vol()
        in_features = self.prepare_input()
        # print("Post im2col:", in_features.size())
        self.data = torch.Tensor()   
        self.data = F.conv2d(in_features, self.weight.view(self.weight.size(0), -1), self.bias.view(self.bias.size(0), -1))
        # Reshape to feature volume 
        self.data = self.data.view(self.data.size(0), self.output_dim[0], self.output_dim[1], self.output_dim[2])
        # print("Reshape:", self.data.size())
        return self
