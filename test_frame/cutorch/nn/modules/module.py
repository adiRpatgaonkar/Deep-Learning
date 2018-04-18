""" NN module class """

from __future__ import print_function

from collections import OrderedDict as OD

import names

class Module(object):
    """ Base class for all nn modules """
    _forward_hooks = OD() # Capture every fprop
    _backward_hooks = OD() # For backprop (Derived from forward graph)
    is_train = False
    is_eval = False

    def __init__(self):
        self._hypers = OD()
        # For capturing connections b/w layers only
        self._forward_graph = None
        self.modules = OD()
        self.param_groups = OD()
        self.gradients = OD()
        self.data = 0

    def __call__(self, *inputs):
        # Check if caller is not an inbuilt module class
        if not type(self).__name__ in names._all:
            # Search for containers if any.
            for name, member in self.__dict__.items():
                # Add container to custom model modules
                if type(member).__name__ in names.containers:
                    if member not in self.modules.values():
                        idx = str(len(self.modules))
                        self._add_module(idx, member)
                        if hasattr(member, "param_groups"): # Sanity check
                            self._add_parameters(member.idx, member.parameters())
        self._add_forward_hooks()
        result = self.forward(*inputs)
        return result

    def __getitem__(self, component, idx=None):
        if component == "base:module":
            return Module._forward_hooks['0']
        elif component == "module":
            if idx in self.modules.keys():
                return self.modules[idx]
            else:
                print("Module doesn't exist.")
                raise KeyError
        elif component == "base:gradients":
            if idx in self.gradients.keys():
                return self.gradients[idx]
            else:
                print("Gradients don't exist.")
                raise KeyError
        elif component == "data":
            try:
                return self._forward_graph[int(idx)].data
            except ValueError:
                print("Could not find module output")

    @staticmethod
    def train():
        Module.is_train = True
        Module.is_eval = False

    @staticmethod
    def eval():
        Module.is_train = False
        Module.is_eval = True

    @staticmethod
    def register_forward_hooks():
        temp = []
        for hook in Module._forward_hooks.values():
            if type(hook).__name__ in names.layers:
                if not hook in temp:
                    temp.append(hook)
        return temp

    @staticmethod
    def forward_stack():
        return Module._forward_hooks

    @staticmethod
    def register_backward_hooks(graph):
        for i, hook in enumerate(graph):
            hook.idx = str(i)
        graph.reverse()
        for hook in graph:
            Module._backward_hooks[hook.idx] = hook
            

    def see_modules(self):
        print("\n" + type(self).__name__, end=" ")
        print("(")
        for idx, module in self.modules.items():
            print(" {}. {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            print("")
        print(")\n")

    def forward(self, *inputs):
        """
        Should be overridden by every subclass module

        Usually:
        :param inputs: Input data/batch
        :return model: last module fproped/prediction
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
            self.gradients[module.idx] = gradients
        # self.update_parameters(lr=0.05)
            
    def set_hyperparameters(self, **kwargs):
        self._hypers = kwargs

    def hypers(self, name):
        if name in self._hypers.keys():
            return self._hypers['lr']
        else:
            print("Wrong hyper-parameter.")
            raise KeyError

    def parameters(self):
        return self.param_groups.values()

    def update_parameters(self, lr, reg=None):
        self._hypers['lr'] = lr
        for module in self._forward_graph:
            for param in module.parameters():
                # if reg is not None and param.tag == 'weight':
                #     # Add regularization gradient contribution
                #     param.gradient += (reg * param.data)
                # Parameter update
                param.data = param.data + (-lr * param.gradient)

    def _add_module(self, idx, module):
        if idx not in self.modules.keys():
            self.modules[idx] = module
            module.idx = idx # May not be absolute
    
    def _add_parameters(self, idx, parameters):
        if parameters not in self.parameters():
            self.param_groups[idx] = parameters

    def _add_forward_hooks(self):
        if self not in Module._forward_hooks.values():
            Module._forward_hooks[str(len(Module._forward_hooks))] = self
