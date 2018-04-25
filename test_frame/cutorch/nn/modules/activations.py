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
        self.input = 0
        self.grad = OrderedDict()
        self.grad['output'] = 0

    def forward(self, in_features):
        if not torch.is_tensor(in_features):
            self.input = in_features.data
        else:
            self.input = in_features
        self.data = f.relu(self.input)
        return self

    def backward(self, gradients):
        """
        :param gradients: gradients from the last 
                          back-propagated layer
        """
        self.grad['output'] = f.gradient_relu(self.data, gradients['output'])
        return self.grad


class Softmax(Module):
    """ Classifier Softmax class """
    def __init__(self):
        super(Softmax, self).__init__()
        # self.parent = None # No use as of now.
        self.idx = -1
        self.input = 0
        self.confidence = 0
        self.prediction = 0
        self.grad = OrderedDict()

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

    def backward(self, targets):
        # Gradient of softmax outputs @ fprop
        self.grad['output'] = f.gradient_softmax(self.data, targets)
        return self.grad

    def predict(self):
        # Return predictions in evaluation mode
        self.confidence, self.prediction = torch.max(self.data, 1)
        self.data = (self.confidence, self.prediction)
        return self
