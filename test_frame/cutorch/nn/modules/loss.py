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
        self.reg_loss = 0
        self.in_model = None

    def forward(self, module, targets):
        """
        :param model: Model object
        :param targets: Targets List
        :return loss: Loss: Scalar
        """
        self.in_model = type(module)._forward_hooks['0']
        self.inputs = module.data
        self.targets = targets
        self.n_log_loss = f.cross_entropy(module.data, targets)
        self.data = f.average_loss(self.n_log_loss)
        # Check if regularization is applicable
#        if 'reg_strength' in (type(module).hyperparameters()).keys():
#            reg = type(module).hyperparameters()['reg_strength']
#            self.data += f.l1_regularization(reg, type(module).parameters())

        if f.nan_check(self.data):
            # Quit if loss is NaN.
            print('Loss is NaN\nExiting ...\n')
            sys.exit(1)
        return self

    def backward(self):
        print(self.in_model)
        return self.in_model.backward(self.targets)
