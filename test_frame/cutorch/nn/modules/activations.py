"""
Activations class
1. ReLU
"""

from __future__ import print_function

from collections import OrderedDict

import torch

from .module import Module
from .. import functionals as f

__dlevel__ = 0


class ReLU(Module):
    """ReLU Activation layer class"""

    def __init__(self):
        super(ReLU, self).__init__()
        self.parent = None
        self.idx = 0
        self.inputs = 0
        self.grad = OrderedDict()
        self.grad['output'] = 0

    def forward(self, in_features):
        if not torch.is_tensor(in_features):
            self.inputs = in_features.data
        else:
            self.inputs = in_features
        self.data = f.relu(self.inputs)
        if __debug__:
            if __dlevel__ == 4:
                print(self.data)
        return self

    def backward(self, gradients):
        """
        :param gradients: gradients from the last 
                          back-propagated layer
        """
        self.grad['output'] = f.gradient_relu(self.data, gradients['output'])
        return self.grad


class Softmax(Module):
    """
    Classifier
    Softmax class
    """

    def __init__(self):
        super(Softmax, self).__init__()
        self.parent = None
        self.idx = 0
        self.inputs = 0
        self.confidence = 0
        self.prediction = 0
        self.grad = OrderedDict()

    def forward(self, in_features):
        # Compute softmax probabilities
        if not torch.is_tensor(in_features):
            self.inputs = in_features.data
        else:
            self.inputs = in_features
        self.data = f.softmax(self.inputs)
        if self.is_train:
            return self
        elif self.is_eval:
            return self.predict()

    def backward(self, targets):
        # Gradient of softmax outputs @ fprop
        self.grad['output'] = f.gradient_softmax(self.data, targets)
        return self.grad

    def predict(self):
        # Return predictions in evaluation mode
        self.confidence, self.prediction = torch.max(self.data, 1)
        return self.confidence, self.prediction
