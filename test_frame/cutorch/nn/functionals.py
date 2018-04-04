import torch


def linear(inputs, weight, bias=None):
    if bias is None:
        return torch.mm(inputs, weight)
    else:
        return torch.addmm(bias, inputs, weight)


def relu(inputs):
    relu_activations = torch.clamp(inputs, min=0)
    if __debug__:
        print(inputs)
        print(relu_activations)
    return relu_activations
