""" NN module class """

from __future__ import print_function

from collections import OrderedDict as OD
from copy import deepcopy

import names


class Module(object):
    """Base class for all nn modules"""
    _forward_hooks = OD()  # Capture every fprop
    _backward_hooks = OD()  # For backprop (Derived from forward graph)
    is_train = False
    is_eval = False

    def __init__(self):
        self._hypers = OD()
        # For capturing connections b/w layers only
        self._forward_graph = OD()
        self.forward_path = OD()
        self._param_graph = OD()
        self._state_dict = OD({'accuracy': 0, 'weight': OD()})
        self.modules = OD()
        self.param_groups = OD()
        self.gradients = OD()
        self.data = 0
        self.results = OD({'accuracy': 0, 'class_performace': []})

    def __call__(self, *inputs):
        # Check if caller is not an inbuilt module class
        if not type(self).__name__ in names.all:
            # Search for containers if any.
            for _, member in self.__dict__.items():
                # Add container to custom model modules
                if type(member).__name__ in names.containers:
                    if member not in self.modules.values():
                        idx = str(len(self.modules))
                        self._add_module(idx, member)
                        if hasattr(member, "param_groups"):  # Sanity check
                            self._add_parameters(member.idx, member.parameters())
        self._add_forward_hooks()
        result = self.forward(*inputs)
        if not type(self).__name__ in names.all:
            self.register_forward_hooks(string=True) 
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
    def forward_stack():
        return Module._forward_hooks

    @staticmethod
    def register_backward_hooks(graph):
        for i, hook in enumerate(graph):
            # Hook => layer module
            hook.idx = str(i)
        graph.reverse()
        for hook in graph:
            Module._backward_hooks[hook.idx] = hook

    def set_state(self, key, value):
        self._state_dict[key] = value

    def get_state(self, key):
        return self._state_dict[key]

    def clean(self, objects):
        for hook in self._forward_graph.values():
            if "input" in objects:
                hook.inputs = None
            if "output" in objects:
                hook.data = 0
            if "gradient" in objects:
                hook.grad_in = 0
                if '_parameters' in hook.__dict__.keys() and len(hook.__dict__['_parameters']) > 0:
                    for param in hook.parameters():
                        param.grad = 0
                self.gradients = OD()
            if "r-graphs" in objects:
                Module._forward_hooks = OD()
                self._backward_hooks = OD()

    def state_dict(self):
        self.clean(["input", "output", "gradient", "r-graphs"])
        return deepcopy(self)

    def see_modules(self):
        print("\n" + type(self).__name__, end=" ")
        print("(")
        for idx, module in self.modules.items():
            print(" ({}) -> {} ".format(idx, type(module).__name__), end="")
            if 'in_features' in module.__dict__ and 'out_features' in module.__dict__:
                print("({}x{})".format(module.in_features, module.out_features), end="")
            if 'in_channels' in module.__dict__ and 'out_channels' in module.__dict__:
                print("({} > {})".format(module.in_channels, module.out_channels), end="")
            print("->",)
        print(")")

    def set_hypers(self, **kwargs):
        self._hypers = kwargs

    def hypers(self, name):
        if name in self._hypers.keys():
            return self._hypers['lr']
        else:
            print("Wrong hyper-parameter.")
            raise KeyError

    def parameters(self, group=False):
        if group:
            return self.param_groups
        else:
            return self._param_graph

    def update_params(self, lr, reg=None):
        self._hypers['lr'] = lr
        for module in self.forward_graph().values():
            for param in module.parameters():
                if reg is not None and param.tag == 'weight':
                    # Add regularization gradient contribution
                    param.grad += (reg * param.data)
                # Parameter update
                param.data = param.data + (-lr * param.grad)

    def _add_module(self, idx, module):
        if idx not in self.modules.keys():
            self.modules[idx] = module
            module.idx = idx  # May not be absolute

    def _add_parameters(self, idx, parameters):
        if parameters not in self.parameters().values():
            self.param_groups[idx] = parameters

    def _add_forward_hooks(self):
        if self not in Module._forward_hooks.values():
            Module._forward_hooks[str(len(Module._forward_hooks))] = self

    def register_forward_hooks(self, string=False):
        # Hook => layer module
        for hook in Module._forward_hooks.values():
            if type(hook).__name__ in names.layers:
                if hook not in self._forward_graph:
                    if string:
                        idx = len(self.forward_path)
                        self.forward_path[str(idx)] = type(hook).__name__
                    elif not string:
                        idx = len(self._forward_graph)
                        self._forward_graph[str(idx)] = hook
                        self._param_graph[str(idx)] = hook.parameters()

    def forward_graph(self):
        return self._forward_graph

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
            self.register_forward_hooks()
            self.register_backward_hooks(self._forward_graph.values()[:])
        gradients = targets  # Alias for targets of classifier
        for module in Module._backward_hooks.values():
            gradients = module.backward(gradients)
            # Store gradients @ current iteration
            # for every module
            # self.gradients[module.idx] = gradients
        # self.update_parameters(lr=0.05)
