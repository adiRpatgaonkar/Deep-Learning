""" Parameters class for the model """


class Parameter:

    def __init__(self, **kwargs):

    	print kwargs
    	if 'weight' in kwargs:
    		module.weight = torch.randn(self.in_features,
                                        self.out_features)
        	module.weight.data *= 0.01 # Conditionalize this
			module._parameters[module] = [module.weight]

        if 'bias' in kwargs:
            module.bias = torch.Tensor(1, out_features).fill_(0)
            module._parameters[module].append(module.bias)
        
        for key, value in kwargs.items():
            self.tag = key
            self.data = value
