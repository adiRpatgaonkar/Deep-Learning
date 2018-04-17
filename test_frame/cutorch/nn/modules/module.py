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
    def __init__(self):
        self._modules = OD()
        self._parameters = OD()
        self._hyperparameters = OD()
        self._gradients = OD()
        self.data = 0
        self.is_train = False
        self.is_eval = False

    def __call__(self, *inputs):
        if self not in Module._forward_hooks.values():
            Module._forward_hooks[str(len(Module._forward_hooks))] = self
        result = self.forward(*inputs)
        return result

    def train(self):
        for member in self.__dict__.values():
            if type(member).__name__ == 'Sequential':
                member._hyperparamters = self._hyperparameters
                member.is_train = True
                member.is_eval = False
                for module in member._modules.values():
                    module.is_train = True
                    module.is_eval = False

    def eval(self):
        for member in self.__dict__.values():
            if type(member).__name__ == "Sequential":
                member.is_train = False
                member.is_eval = True
                for module in member._modules.values():
                    module.is_train = False
                    module.is_eval = True

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
            self.register_backward_hooks(self._forward_graph)
        gradients = targets # Alias for targets of classifier
        for module in Module._backward_hooks.values():
            gradients = module.backward(gradients)
            # Store gradients @ current iteration
            # for every module
            self.gradients[module] = gradients
            
    def register_forward_hooks(self):
        temp = []
        for hook in Module._forward_hooks.values():
            print(hook)
            if type(hook).__name__ in Module._layers:
                temp.append(hook)
        return temp
    
    def register_backward_hooks(self, graph):
        for i, hook in enumerate(graph):
            hook.idx = str(i)
            print(hook, hook.idx)
        graph.reverse()
        for hook in graph:
            print(hook)
            Module._backward_hooks[hook.idx] = hook
        print(Module._backward_hooks)
            
    def set_hyperparameters(self, **kwargs):
        self._hyperparameters = kwargs

    def hyperparameters(self):
        return self._hyperparameters

    def parameters(self):
        # If parameters are not added to the model,
        # add 'em right away.
        if not self._parameters:
            for member in self.__dict__.values():
                print(type(member))
                if type(member).__name__ == 'Sequential':
                    if member not in self._parameters:
                        self._parameters[member] = member.parameters()
        return self._parameters

    def update_parameters(self):
        return self.update_parameters()

    def _add_module(self, idx, module):
        self._modules[idx] = module
        module.idx = idx
    
    def _add_parameters(self, idx, module):
        self._parameters[idx] = module._parameters

    # def _add_forward_hooks(self, module):
    #     if module not in _forward_hooks.values():
    #         _forward_hooks[str(len(_forward_hooks.keys()))] = module

    def _add_backward_hooks(self):
        """
        Not required by the Sequential Container
        """
        raise NotImplementedError

    def see_modules(self):
        print("\n" + type(self).__name__, end=" ")
        print("(")
        for idx, module in self._modules.items():
            print(" {}. {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            print("")
        print(")\n")
