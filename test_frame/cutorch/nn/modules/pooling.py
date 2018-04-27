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
        batch_im = torch.Tensor()
        for image in in_features: 
            batch_im = torch.cat((batch_im, F.im2col(image, self.spatial_extent, self.stride, task="pooling").unsqueeze_(0)), 0)
        return batch_im

    def forward(self, in_features):
        """ Pooling op """
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        print("Input to max_pool2d layer:", self.input.size())
        self.create_output_vol(self.input)
        in_features = self.prepare_input(self.input) # post im2col
        #print("Post im2col:", in_features.size())
        self.data, self.max_track = F.max_pool2d(in_features)
        #print("Post_max_pool", self.data) 
        self.data = self.data.view(self.data.size(0), self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return self
