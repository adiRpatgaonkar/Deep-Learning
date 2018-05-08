""" 
Conv layers' class
1. Conv2d
"""

from __future__ import print_function

from collections import OrderedDict

import torch
import numpy as np

from cutorch.nn.parameter import Parameter
from .module import Module
from .. import functionals as F


class Conv2d(Module):
    """2D Conv layer class"""

    def __init__(self, channels, kernels, kernel_size=3, pad=0, stride=1, bias=True, beta=0.01):
        super(Conv2d, self).__init__()
        # Layer construct check
        assert pad >= 0, ("Invalid padding value. Should be >= 0")
        assert stride > 0, ("Invalid stride. Should be > 0")
        self.idx = -1
        self.channels = channels
        self.kernel_size = kernel_size
        self.kernels, self.padding, self.stride = kernels, pad, stride
        self.input = None  # TODO:CLEAN
        self.data = 0  # TODO:CLEAN
        self.output_dim = [0, 0, 0]
        self.batch_ims = None # im2col data.  # TODO:CLEAN
        # Parameters' creation
        self._parameters = []
        self.weight = Parameter(weight=beta*torch.randn(self.kernels, channels, self.kernel_size, self.kernel_size),
                                require_gradient=True)
        if bias is True:
            self.bias = Parameter(bias=torch.Tensor(self.kernels, 1, 1, 1).fill_(1),
                                  require_gradient=True)  # print(self.biases)
        # Gradients' creation
        self.grad = OrderedDict()  # TODO:CLEAN
        if self.weight.require_gradient:
            self.grad['weight'] = 0
        if bias and self.bias.require_gradient:
            self.grad['bias'] = 0
        self.grad['in'] = 0  # CLEAN
         
        # Finish param setup
        self.init_param_setup()

    def parameters(self):
        return self._parameters

    def init_param_setup(self):
        self._parameters.append(self.weight)
        if 'bias' in self.__dict__:
            self._parameters.append(self.bias)

    def create_output_vol(self):
        """ Create output volume """
        # Check & setup input tensor dimensions
        # Input should be a batch (4D) or 3D tensor
        assert self.input.dim() in (3, 4), ("Input tensor should be 3D or 4D")
        if self.input.dim() == 3: # Convert to 4D tensor
            self.input = torch.unsqueeze(self.input, 0)
        self.N, self.C, self.H, self.W = self.input.size()
        assert self.C == self.channels, \
               "Input channels should be {}. Got {} channels".format(self.channels, self.C)
        self.output_dim[0] = self.kernels
        self.output_dim[1] = ((self.H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.output_dim[2] = ((self.W - self.kernel_size + 2 * self.padding) / self.stride + 1)

    def prepare_input(self):
        """ 
        Prepare in features
        1. Padding
        2. im2col
        """
        # 1. Padding: self.input should be 4D
        if self.padding > 0:
            self.input = F.pad_image(self.input, self.padding)
            if type(self.input) is np.ndarray: # If numpy array, convert to tensor before conv op.
                self.input = torch.from_numpy(self.input)
        return F.im2col(self.input, self.kernel_size, self.stride)
 
    def forward(self, in_features):
        """ Convolution op (Auto-correlation i.e. no kernel flipping) """
        # Check for input tensor
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features 
        self.create_output_vol()
        print("Input to conv layer:", self.input.size())
        self.input = self.prepare_input() # im2col'ed input
        #print("Post im2col:", self.input.size())
        print(self.input.size(), self.weight.data.view(self.weight.data.size(0), -1).size(), 
        self.bias.data.size())
        self.data = F.conv_2d(self.input, self.weight.data.view(self.weight.data.size(0), -1), 
                              self.bias.data.view(self.bias.data.size(0), -1))
        # Reshape to the o/p feature volume
        self.data = self.data.view(self.N, self.output_dim[0], self.output_dim[1], 
                                   self.output_dim[2])
        # print("Reshaped:", self.data.size())
        # Clean
        del in_features
        return self

    def backward(self, gradients):
        # gradients['in'] are actually output gradients
        # grad['in'] are actual input gradients
        
        if self.bias.require_gradient:
            self.grad['bias'] = F.grad_conv2d_bias(gradients['in'])
            self.grad['bias'] = self.grad['bias'].unsqueeze(1).unsqueeze(1)
            self.bias.gradient = self.grad['bias'] 

        if self.weight.require_gradient:
            self.grad['weight'], cache = F.grad_conv2d_weight(self.input, gradients['in'])   
            # Reshape to the size of the layer's kernels
            self.grad['weight'] = self.grad['weight'].view(self.weight.data.size())
        self.idx = '1'
        if self.idx == '0':
            # No gradients required for input layer (idx == 0)
            self.grad['in'] = torch.Tensor(self.input.size())
        else:
            self.grad['in'] = F.grad_conv2d(cache, self.weight.data, gradients['in']) 
        # De-im2col grad ins i.e. F.col2im(self.grad['in'])
            self.grad['in'] = F.col2im(self.grad['in'], (self.N, self.C, self.H, self.W),
                                          self.kernel_size, self.stride)
        # Clean
        del gradients, cache
        return self.grad

