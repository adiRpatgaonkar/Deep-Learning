""" Loss' classes """

from __future__ import print_function

import sys

from .module import Module
from .. import functionals as f


class CrossEntropyLoss(Module):
    """ Calculate cross-entropy loss """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.input = 0
        self.targets = 0
        self.n_log_loss = 0
        self.reg_loss = 0
        self.in_model = None

    def forward(self, module, targets):
        """
        :param model: Model object
        :param targets: Targets List
        :return loss: Loss: Scalar
        """
        # Capture model from the 1st
        # forward call.
        self.in_model = module["base:module"]
        self.input = module.data
        self.targets = targets
        self.n_log_loss = f.cross_entropy(module.data, targets)
        self.data = f.average_loss(self.n_log_loss)

        if f.nan_check(self.data):
            # Quit if loss is NaN.
            sys.exit('Loss is NaN\nExiting ...\n')
        del module
        return self

    def backward(self):
        return self.in_model.backward(self.targets)
