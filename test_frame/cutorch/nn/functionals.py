import torch

global DEBUG
DEBUG = False

def linear(inputs, weight, bias=None):
	if DEBUG:
		pass
	if bias is None:
		return torch.mm(inputs, weight)
	else:
		return torch.addmm(bias, inputs, weight)

def relu(inputs):
    relu_activations = torch.clamp(inputs, min=0)
    if DEBUG:
    	print(inputs)
    	print(relu_activations)
    return relu_activations