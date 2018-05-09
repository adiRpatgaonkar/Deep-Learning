"""
Activations class
1. ReLU
"""

from __future__ import print_function

from collections import OrderedDict

import torch

from .module import Module
from .. import functionals as f


class ReLU(Module):
    """ ReLU Activation layer class """
    def __init__(self):
        super(ReLU, self).__init__()
        self.idx = -1
        self.input = None  # CLEAN
        self.data = 0  # CLEAN
        # Gradients' creation
        self.grad_in = None  # CLEAN

    def forward(self, in_features):
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        self.data = f.relu(self.input)
        return self

    def backward(self, grad_out):
        """
        :param gradients: gradients from the last 
                          back-propagated layer
        """
        self.grad_in = f.gradient_relu(self.data, grad_out)
        return self.grad_in


class Softmax(Module):
    """ Classifier Softmax class """
    def __init__(self):
        super(Softmax, self).__init__()
        # self.parent = None # No use as of now.
        self.idx = -1
        self.input = None # CLEAN
        self.data = 0 # CLEAN
        self.confidence = 0
        self.prediction = 0
        self.grad_in = None  # CLEAN
        
    def forward(self, in_features):
        # Compute softmax probabilities
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        self.data = f.softmax(self.input)
        if Module.is_train:
            return self
        elif Module.is_eval:
            return self.predict()
        else:
            raise AssertionError("Mode not specified: (train/eval)")

    def backward(self, targets):
        # targets: LongTensor of labels
        # grad['in'] are actual input gradients
        # Gradient of softmax outputs @ fprop
        self.grad_in = f.gradient_softmax(self.data, targets)
        return self.grad_in

    def predict(self):
        # Return predictions in evaluation mode
        self.confidence, self.prediction = torch.max(self.data, 1)
        self.data = (self.confidence, self.prediction)
        return self

