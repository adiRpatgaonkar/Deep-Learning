from __future__ import print_function


class Optimize:
    """Schedules L.R and saves the optimum parameters"""

    def __init__(self, parameters, config=None, max_epochs=None, lr=None, lr_decay=None):
        # Store model instance via the parameters method
        print("Training with SGD:\n")
        self.model = parameters.im_self
        if config is not None:
            self.lr0 = config['lr']
            self.max_epochs = config['max_epochs']
            self.curr_epoch = config['start_epoch']
        elif (max_epochs is not None and 
            lr is not None and 
            lr_decay is not None):
            self.max_epochs = max_epochs
            self.curr_iter = 0
            self.lr0 = lr
            self.lr = self.lr0
            self.lr_decay = lr_decay

    def step(self):
        for mem in self.model.__dict__.values():
            if type(mem).__name__ == 'Sequential':
                if self.curr_iter % 50 == 0:
                    print("L.R:[{:.5f}]".format(self.lr), end=" ")
                mem.update_parameters(self.lr)
                self.curr_iter += 1
                self.time_decay()

    def time_decay(self):
        self.lr = self.lr0 / (1 + self.lr_decay * self.curr_iter)

    def step_decay(self, decay_after=5, drop=0.5):
        if self.curr_epoch % decay_after == 0:
            self.lr *= drop
        pass

    def exp_decay(self):
    #    self.lr = (self.lr0 * math.exp(-self.lr_decay * self.curr_epoch))
        pass

    def set_optim_param(self):
        # Check if you've got the best params via accuracies
        pass

    def clear_gradients(self):
        pass