""" NN module class """

from __future__ import print_function

from collections import OrderedDict as OD

__dlevel__ = 0


class Module(object):
    """ Base class for all nn modules """
    _forward_hooks = OD()
    _backward_hooks = OD()
    _containers = {"Sequential"}
    _layers = {"Linear", "ReLU", "Softmax"}
    is_train = False
    is_eval = False

    def __init__(self):
        self._modules = OD()
        self._parameters = OD()
        self._hypers = OD()
        self._forward_graph = None
        self.gradients = OD()
        self.data = 0

    def __call__(self, *inputs):
        self._add_forward_hooks()
        result = self.forward(*inputs)
        return result

    @staticmethod
    def train():
        Module.is_train = True
        Module.is_eval = False

    @staticmethod
    def eval():
        Module.is_train = False
        Module.is_eval = True

    def forward(self, *inputs):
        """
        Should be overridden by every subclass module

        Usually:
        :param inputs: Input data/batch
        :return model: Post fprop
        """
        raise NotImplementedError

    def backward(self, targets):
        if not Module._backward_hooks:
            self._forward_graph = self.register_forward_hooks()
            self.register_backward_hooks(self._forward_graph[:])
        gradients = targets  # Alias for targets of classifier
        for module in Module._backward_hooks.values():
            gradients = module.backward(gradients)
            # Store gradients @ current iteration
            # for every module
            self.gradients[module] = gradients
        # self.update_parameters(lr=0.05)
            
    @staticmethod
    def register_forward_hooks():
        temp = []
        for hook in Module._forward_hooks.values():
            if type(hook).__name__ in Module._layers:
                temp.append(hook)
        return temp
    
    @staticmethod
    def register_backward_hooks(graph):
        for i, hook in enumerate(graph):
            hook.idx = str(i)
        graph.reverse()
        for hook in graph:
            Module._backward_hooks[hook.idx] = hook
            
    def set_hyperparameters(self, **kwargs):
        self._hypers = kwargs

    def hypers(self, name):
        if name in self._hypers.keys():
            return self._hypers['lr']
        else:
            print("Wrong hyper-parameter.")
            raise KeyError

    def parameters(self):
        # If parameters are not added to the model,
        # add 'em right away.
        for hook in Module._forward_hooks.values():
            if type(hook).__name__ in Module._layers:
                if hook._parameters:
                    self._parameters[str(len(self._parameters))] = hook._parameters
        return self._parameters.values()

    def update_parameters(self, lr, reg=None):
        self._hypers['lr'] = lr
        for module in self._forward_graph:
            for param in module._parameters:
                if reg is not None and param.tag == 'weight':
                    # Add regularization gradient contribution
                    param.gradient += (reg * param.data)
                # Parameter update
                param.data = param.data + (-lr * param.gradient)

    def _add_module(self, idx, module):
        self._modules[idx] = module
        module.idx = idx
    
    def _add_parameters(self, idx, module):
        self._parameters[idx] = module._parameters

    def _add_forward_hooks(self):
        if self not in Module._forward_hooks.values():
            Module._forward_hooks[str(len(Module._forward_hooks))] = self

    def see_modules(self):
        print("\n" + type(self).__name__, end=" ")
        print("(")
        for idx, module in self._modules.items():
            print(" {}. {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            print("")
        print(")\n")
