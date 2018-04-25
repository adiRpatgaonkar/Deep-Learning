from __future__ import print_function

import torch

from .optimizer import Optimizer
from ..utils.model_store import save


# TODO: Momentum (SGD)

# Base optimizer class
class SGD(Optimizer):
    """Schedules L.R and saves the optimum parameters"""

    def __init__(self, model, lr, lr_decay, momentum, reg=None):
        super(SGD, self).__init__(model, lr, lr_decay, reg)
        # Capture model (Alias)
        self.v = torch.FloatTensor([0])
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
        for module in self.model.forward_graph().values():
            for param in module.parameters():
                if self.reg is not None and param.tag == 'weight':
                    # Add regularization gradient contribution
                    param.gradient += (self.reg * param.data)
                # Momentum update
                if self.v.dim() > 1 and self.v.size() == param.gradient.size():
                    self.v = (self.mu * self.v) - (self.lr * param.gradient)
                    # Parameter update
                    param.data = param.data + self.v