from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F

class MaxPool2d(Module):
    """Max/Mean pooling layer class"""

    def __init__(self, f, stride=None):
        super(MaxPool2d, self).__init__()
        if not stride:
            self.stride = f
        else:
            self.strides = stride
        self.spatial_extent = f
    
    def create_output_vol(self, input):
        """ Create output volume """
        self.height, self.width = input.size()[2:]
        self.output_dim = [0, 0, 0] # For a single image
        if input.dim() == 4:
            self.output_dim[0] = input.size(1)
        elif input.dim() == 3:
            self.output_dim[0] = input.size(0)
        self.output_dim[1] = ((self.width - self.spatial_extent) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.spatial_extent) / self.stride) + 1

    def prepare_input(self, in_features):
        """ Prepare in features """
        if in_features.dim() == 3: # For a single image
            in_features = torch.unsqueeze(in_features, 0)
        # im2col operation
        # for image in in_features:
        #     in_features = F.im2col(image, self.spatial_extent, self.stride)
        # return in_features
        batch_im = torch.Tensor()
        for image in in_features:
            batch_im = torch.cat((batch_im, F.im2col(image, self.spatial_extent, self.stride).unsqueeze_(0)), 0)
        return batch_im

    def forward(self, in_features):
        """ Pooling op """
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        # print("Input to max_pool2d layer:", self.input.size())
        self.create_output_vol(self.input)
        in_features = self.prepare_input(self.input) # post im2col
        # print("Post im2col:", in_features.size())
        self.data, self.max_track = F.max_pool2d(in_features)
        self.data, self.max_track = torch.Tensor(), torch.Tensor()
        for image in in_features:
            cache_data, cache_max_track  = F.max_pool2d(image)
            self.data = torch.cat((self.data, cache_data.unsqueeze_(0)), 0)
            self.max_track = torch.cat((self.max_track, (cache_max_track.float()).unsqueeze_(0)), 0)
        # print("Post max_pool2D:", self.data.size())
        self.data.resize_(self.data.size(0), self.output_dim[0], self.output_dim[1], self.output_dim[2])
        # print("Resized:", self.data.size())
        return self
