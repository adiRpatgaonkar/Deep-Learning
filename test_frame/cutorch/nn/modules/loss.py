""" Loss' classes """

from __future__ import print_function

import sys

from .module import Module
from .. import functionals as f


class CrossEntropyLoss(Module):
    """
    Calculate cross-entropy loss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.inputs = 0
        self.targets = 0
        self.n_log_loss = 0
        self.data = 0
        self.out_model = None

    def forward(self, model, targets):
        """
        :param model: model object
        :param targets: Targets List
        :return loss: Loss: Scalar
        """
        self.out_model = model
        # print(type(self.out_model))
        self.inputs = model.data
        self.targets = targets
        self.n_log_loss = f.cross_entropy(model.data, targets)
        self.data = f.average_loss(self.n_log_loss)
        if f.nan_check(self.data):
            # Quit if loss is NaN.
            print('Loss is NaN\nExiting ...\n')
            sys.exit(1)
        return self

    def backward(self):
        return self.out_model.backward(self.targets)
