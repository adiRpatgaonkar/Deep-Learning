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

    def __init__(self, f, stride=None):
        super(MaxPool2d, self).__init__()
        # Layer construct check
        assert f >= 2, ("Invalid padding value. Should be >= 2")
        if stride:
            assert stride > 0, ("Invalid stride. Should be > 0")
        self.idx = -1
        self.kernel_size = f
        # If stride is not given, set equal to spatial extent
        if stride is None:
            self.stride = f
        else:
            self.stride = stride
        self.input = None  # TODO:CLEAN
        self.data = 0  # TODO:CLEAN
        self.height = self.width = 0
        self.output_dim = [0, 0, 0]  # For a single image
        self.batch_ims = None # im2col data.  # TODO:CLEAN
        # Gradients' creation
        self.grad = OrderedDict()
        self.grad['output'] = 0
    
    def create_output_vol(self):
        """ Create output volume """
        # Check & setup input tensor dimensions
        # Input should be a batch (4D) or 3D tensor
        assert self.input.dim() in (3, 4), ("Input tensor should be 3D or 4D")
        if self.input.dim() == 3: # Convert to 4D tensor
            self.input = torch.unsqueeze(self.input, 0)
        self.height, self.width = self.input.size()[2:]
        self.output_dim[0] = self.input.size(1)
        self.output_dim[1] = ((self.width - self.kernel_size) / self.stride) + 1
        self.output_dim[2] = ((self.height - self.kernel_size) / self.stride) + 1

    def prepare_input(self):
        """ Prepare in features """
        # 1. im2col operation (Batch op) 
        return F.im2col(self.input, self.kernel_size, self.stride, task="pool") 

    def forward(self, in_features):
        """ Pooling op """
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features   
        self.create_output_vol()
        N = self.input.size(0)
        print("Input to max_pool2d layer:", self.input.size())
        self.input = self.prepare_input() # im2col'ed input
        print("Post im2col:", self.input.size())
        self.data, self.max_track = F.max_pool2d(self.input)
        print("Post_max_pool:", self.data.size()) 
        self.data = self.data.view(N, self.output_dim[0], 
                                   self.output_dim[1], self.output_dim[2])
        # print("Reshaped:", self.data.size())
        # Clean
        del in_features
        return self

    def backward(self):
        pass
