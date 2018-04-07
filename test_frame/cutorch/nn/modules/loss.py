""" Loss' classes """

from __future__ import print_function

from .module import Module
from .. import functionals as f


class CrossEntropyLoss(Module):
    """
    Calculate cross-entropy loss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.n_log_loss = 0
        self.data = 0

    def forward(self, inputs, targets):
        """
        :param inputs: Softmax probabs Tensor
        :param targets: Targets List
        :return loss: Loss: Scalar
        """
        self.n_log_loss = f.cross_entropy(inputs, targets)
        self.data = f.average_loss(self.n_log_loss)
        return self

    def backward(self):
        pass
