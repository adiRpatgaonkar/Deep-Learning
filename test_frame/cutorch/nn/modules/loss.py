""" Loss' classes """

from __future__ import print_function

import torch

from .module import Module
from .. import functionals as f


class CrossEntropyLoss(Module):
    """
    Calculate cross-entropy loss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        :param inputs: Softmaxed Tensor
        :targets targets: Targets List
        :return loss: Loss: Scalar
        """
        self.n_log_loss = f.cross_entropy(inputs, targets)
        self.loss = f.average_loss(self.n_log_loss)
        return self.loss

