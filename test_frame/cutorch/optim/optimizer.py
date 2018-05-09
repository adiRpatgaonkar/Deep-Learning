from __future__ import print_function
from collections import OrderedDict as OD
from copy import deepcopy

from ..utils.model_store import save


# TODO: Momentum (SGD)

# Base optimizer class
class Optimizer(object):
    """Schedules L.R and saves the optimum parameters"""

    def __init__(self, model, lr, lr_decay=0, reg=None):
        # Store model instance via the parameters method
        assert lr >= 0, "Learning rate should be >= 0"
        assert lr_decay >= 0, "L.R decay should be >= 0"
        if reg:
            assert reg >= 0, "Reg strength should be >= 0"
        print("#StochasticGradientDescent:")
        # Capture model (Alias)
        self.model = model
        self.state = OD()
        self.curr_iter = 0
        self.lr0 = lr  # Set base L.R.
        self.lr = self.lr0
        self.lr_decay = lr_decay
        self.reg = reg
        model.set_hypers(lr=self.lr, lr_decay=self.lr_decay, 
                                  reg=self.reg)

    # Stepper
    def step(self):
        self.model.update_params(lr=self.lr, reg=self.reg)
        self.curr_iter += 1
        if self.lr_decay != 0:
            self.lr = self.time_decay()
        self.model.clean(["input", "output"])

    # Clear Grad
    def zero_grad(self):
        self.model.clean(["gradient"])

    # Select/store best model state/parameters
    def check_model(self, select=False, store=False, name="model.pkl"):
        # Check if you've got the best params via accuracies
        if select:
            print("\nChecking model results ...", end=" ")
            if self.model.results['accuracy'] > self.model.get_state('accuracy'):
                self.model.set_state('accuracy', self.model.results['accuracy'])
                # self.state['weights'] = deepcopy(self.model.parameters()) # Saves params (w & b)
                # self.state['gradients'] = deepcopy(self.model.gradients)
                self.state['model'] = self.model.state_dict()
            print("done.")
        if store:
            try:
                self.state['model'].clean(["input", "output", "gradient"])
            except KeyError:
                print("No best model state saved. Not saving " + name)
                return
            print("Saving model object ...")
            save(self.state['model'], name + ".pkl")
            # print("Saving weights ...")
            # save(self.state['weights'], name + "_weights.pkl")
            # print("Saving gradients ...")
            # save(self.state['gradients'], name + "_grads.pkl")

    # ++++ L.R. Schedulers ++++ #
    def time_decay(self):
        self.lr = self.lr0 / (1 + self.lr_decay * self.curr_iter)
        return self.lr

    def step_decay(self, decay_after=5, drop=0.5):
        if self.curr_iter % decay_after == 0:
            self.lr *= drop
        pass

    def exp_decay(self):
        #  self.lr = (self.lr0 * math.exp(-self.lr_decay * self.curr_iter))
        pass
