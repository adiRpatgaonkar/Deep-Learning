from __future__ import print_function
from collections import OrderedDict as OD
from copy import deepcopy

from .optimizer import Optimizer
from ..utils.model_store import save


# TODO: Momentum (SGD)

# Base optimizer class
class SGD(Optimizer):
    """Schedules L.R and saves the optimum parameters"""

    def __init__(self, parameters, lr, momentum, reg=None):
        super(SGD, self).__init__()
        print("#StochasticGradientDescent with momentum:")
        # Capture model (Alias)
        self.model = parameters.im_self
        self.parameters = parameters
        self.v = 0
        self.mu = momentum
        self.model.set_hyperparameters(momentum=self.mu)

    # Stepper
    def step(self):
        self.update_parameters()
        self.curr_iter += 1
        self.lr = self.time_decay()
        self.model.clean(["input", "output"])

    def update_parameters(self):
        self.model._hypers['lr'] = self.lr
        for param in self.parameters:
            if self.reg is not None and param.tag == 'weight':
                # Add regularization gradient contribution
                param.gradient += (self.reg * param.data)
            # Momentum update
            self.v = self.mu * self.v - self.lr * param.gradient
            # Parameter update
            param.data = param.data + self.v