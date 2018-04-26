from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as f

class Conv2d(Module):
    """2D Conv layer class"""

    def __init__(self, channels, kernels, kernel_size=3, pad=0, stride=1):
        super(Conv2d, self).__init__()
        self.depth = channels
        self.kernel_size = kernel_size
        self.kernels, self.padding, self.stride = kernels, pad, stride
        self.feature_map_volume = torch.zeros(0, 0)

        # Layer construct check
        if pad < 0:
            raise ValueError("Invalid padding value. Should be >= 0")
        if stride <= 0:
            raise ValueError("Invalid padding value. Should be > 0")

    def create_output_vol(self, in_features):
        """ Create output volume """
        self.height, self.width = in_features.size()[2:]
        # Setup weights(kernels).
        self.weight = 0.01 * torch.randn(self.kernels, self.depth, self.kernel_size, self.kernel_size)
        self.bias = torch.ones(self.kernels, 1, 1, 1)  # print(self.biases)
        self.output_dim = [0, 0, 0] # For a single image
        self.output_dim[0] = self.kernels
        self.output_dim[1] = ((self.width - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        # print("Out dims:", self.output_dim, "Feature map:", self.feature_map_volume.size(), "Weights:", self.weight.size())

    def prepare_input(self, in_features):
        """ Prepare in features """
        # 1. Padding
        if self.padding > 0:
            in_features = f.pad_image(in_features, self.padding)
            if torch.is_tensor(in_features):
                in_features = in_features.numpy()
        # 2. im2col operation
        if type(in_features) is np.ndarray: # If numpy array convert to tensor before conv op.
            in_features = torch.from_numpy(in_features) # TODO: Enable cuda compatibilty
        if in_features.dim() == 3: # For single image
            in_features = torch.unsqueeze(in_features, 0)
        batch_im = torch.Tensor()
        for image in in_features:
            batch_im = torch.cat((batch_im, f.im2col(image, self.kernel_size, self.stride).unsqueeze_(0)), 0)
        return batch_im

    def forward(self, in_features):
        """ Convolution op (Auto-correlation i.e. no flipping of kernels)"""
        # Check foe tensor input
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        # print("Input to conv layer:", self.input.size())
        self.create_output_vol(self.input)
        in_features = self.prepare_input(self.input)
        # print("Post im2col:", in_features.size())
        self.data = torch.Tensor()
        for image in in_features:
            self.data = torch.cat((self.data, f.conv2d(image, self.weight.view(self.weight.size(0), -1)).unsqueeze_(0)), 0)
        # print("Post conv2d:", self.data.size())
        self.data.resize_(self.data.size(0), self.output_dim[0], self.output_dim[1], self.output_dim[2])
        # print("Resized:", self.data.size())
        return self
