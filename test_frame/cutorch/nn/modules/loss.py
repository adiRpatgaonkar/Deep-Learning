""" Loss' classes """

from __future__ import print_function

import sys

from .module import Module
from .. import functionals as f


class CrossEntropyLoss(Module):
    def __init__(self):
        """Calculate cross-entropy loss"""
        super(CrossEntropyLoss, self).__init__()
        self.inputs = 0
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
        self.inputs = module.data
        self.targets = targets
        self.n_log_loss = f.cross_entropy(module.data, targets)
        self.data = f.average_loss(self.n_log_loss)
        # Check if regularization is applicable
        #self.data += f.l1_regularization(0.001, self.in_model.parameters())

        if f.nan_check(self.data):
            # Quit if loss is NaN.
            print('Loss is NaN\nExiting ...\n')
            sys.exit(1)
        return self

    def backward(self):
        return self.in_model.backward(self.targets)
