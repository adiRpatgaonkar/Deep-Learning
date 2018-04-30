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
        self.spatial_extent = f
        # If stride is not given, set equal to spatial extent
        if stride is None:
            self.stride = f
        else:
            self.stride = stride
        # Layer construct check
        if f < 2:
            raise ValueError("Invalid padding value. Should be >= 2")
        if stride and stride <= 0:
            raise ValueError("Invalid padding value. Should be > 0")
    
    def create_output_vol(self):
        """ Create output volume """
        # Check & setup input tensor dimensions
        if self.input.dim() != 4:  # For batch image tensor
            if self.input.dim() == 3:  # For a 3D image tensor
                self.input = torch.unsqueeze(self.input, 0)
            else:
                raise ValueError("Input tensor should be 3D or 4D")
        self.height, self.width = self.input.size()[2:]
        self.output_dim = [0, 0, 0] # For a single image
        self.output_dim[0] = self.input.size(1)
        self.output_dim[1] = ((self.width - self.spatial_extent) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.spatial_extent) / self.stride) + 1

    def prepare_input(self):
        """ Prepare in features """
        # 1. im2col operation (One image @ a time.)
        batch_ims = torch.Tensor()
        for image in self.input: 
            batch_ims = torch.cat((batch_ims, F.im2col(image, self.spatial_extent, self.stride, task="pooling").unsqueeze_(0)), 0)
        return batch_ims

    def forward(self, in_features):
        """ Pooling op """
        # Check for tensor input
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        print("Input to max_pool2d layer:", self.input.size())
        self.create_output_vol()
        self.input = self.prepare_input() # im2col'ed input
        #print("Post im2col:", self.input.size())
        self.data, self.max_track = F.max_pool2d(self.input)
        #print("Post_max_pool", self.data) 
        self.data = self.data.view(self.data.size(0), self.output_dim[0], 
                                   self.output_dim[1], self.output_dim[2])
        # print("Reshaped:", self.data.size())
        return self

    def backward(self):
        pass