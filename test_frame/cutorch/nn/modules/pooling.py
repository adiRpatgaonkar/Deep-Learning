""" 
Pooling layers' class
1. MaxPool2d
"""
from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F

class MaxPool2d(Module):
    """Max/Mean pooling layer class"""

    def __init__(self, spatial_extent, stride=None):
        super(MaxPool2d, self).__init__()
        # Layer construct check
        assert spatial_extent >= 2, ("Invalid padding value. Should be >= 2")
        if stride:
            assert stride > 0, ("Invalid stride. Should be > 0")
        self.idx = -1
        self.kernel_size = spatial_extent
        # If stride is not given, set equal to spatial extent
        if stride is None:
            self.stride = spatial_extent
        else:
            self.stride = stride
        self.input = None  # TODO:CLEAN
        self.data = 0  # TODO:CLEAN
        self.N = self.H = self.W = 0
        self.output_dim = [0, 0, 0]  # For a single image
        # Grad input
        self.grad_in = None
    
    def create_output_vol(self):
        """ Create output volume """
        # Check & setup input tensor dimensions
        # Input should be a batch (4D) or 3D tensor
        assert self.input.dim() in (3, 4), ("Input tensor should be 3D or 4D")
        if self.input.dim() == 3: # Convert to 4D tensor
            self.input = torch.unsqueeze(self.input, 0)
        self.N, self.C, self.H, self.W = self.input.size()
        self.output_dim[0] = self.C  # Depth remains the same
        self.output_dim[1] = ((self.H - self.kernel_size) / self.stride) + 1
        self.output_dim[2] = ((self.W - self.kernel_size) / self.stride) + 1

    def prepare_input(self):
        """ Prepare in features """
        # 1. im2col operation (Batch op) Different than conv's im2col
        return F.im2col(self.input.view(self.N * self.C, 1, self.H, self.W), 
                        self.kernel_size, self.stride)

    def forward(self, in_features):
        """ Pooling op """
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features   
        self.create_output_vol()
        N = self.input.size(0)
        #print("Input to max_pool2d layer:", self.input.size())
        self.input = self.prepare_input() # im2col'ed input
        #print("Post im2col:", self.input.size())
        self.data, self.max_track = F.max_pool2d(self.input)
        #print("Post_max_pool:", self.data.size()) 
        self.data = self.data.view(N, self.output_dim[0], 
                                   self.output_dim[1], self.output_dim[2])
        #print("Reshaped:", self.data.size())
        return self

    def backward(self, grad_out):
        self.grad_in = F.grad_maxpool2d(self.input.size(), self.max_track, grad_out)
        self.grad_in = F.col2im(self.grad_in, (self.N * self.C, 1, self.H, self.W),
                                      self.kernel_size, self.stride)
         # Different than conv's col2im
        self.grad_in = self.grad_in.view(self.N, self.C, self.H, self.W)
        return self.grad_in
                                    
