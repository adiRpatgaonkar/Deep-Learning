from __future__ import print_function
from copy import deepcopy

class SGD:
    """Schedules L.R and saves the optimum parameters"""

    #TODO: Momentum
    def __init__(self, model, lr, lr_decay=0, reg_strength=0):
        # Store model instance via the parameters method
        print("#StochasticGradientDescent:")
        # Capture model (Alias)
        self.model = model
        # TODO : Load from config file
        self.curr_iter = 0
        self.lr0 = lr # Set base L.R.
        self.lr = self.lr0
        self.lr_decay = lr_decay
        model.set_hyperparameters(lr=self.lr,
                                  lr_decay=self.lr_decay)

    def step(self):
        self.model.update_parameters(lr=self.lr)
        self.curr_iter += 1
        self.lr = self.time_decay()
        self.set_best_params()

    def time_decay(self):
        self.lr = self.lr0 / (1 + self.lr_decay * self.curr_iter)
        return self.lr

    def step_decay(self, decay_after=5, drop=0.5):
        if self.curr_iter % decay_after == 0:
            self.lr *= drop
        pass

    def exp_decay(self):
        #  self.lr = (self.lr0 * math.exp(-self.lr_decay * self.curr_epoch))
        pass

    def set_best_params(self):
        # Check if you've got the best params via accuracies
        if self.model.results['accuracy'] > self.model.get_state('accuracy'):
            self.model.set_state('accuracy', self.model.results['accuracy'])
            self.model.set_state('best_params' , deepcopy(self.model.parameters(graph=True)))

    def clear_gradients(self):
        pass
