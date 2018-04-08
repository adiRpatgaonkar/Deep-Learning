from __future__ import print_function

import random

import torch

__dlevel__ = 0


def flatten(data):
    """
    Flatten data Tensor
    :param data: 2D Tensor
    :return: flattened Tensor(1D Tensor)
    """
    num_examples = 1
    if data.dim() == 3:
        num_examples = 1
        data = data.unsqueeze(0)
    elif data.dim() == 4:
        num_examples = data.size(0)

    return data.view(num_examples, -1)


def linear(inputs, weight, bias=None):
    """
    out = Ax + b
    :param inputs: Input features Tensor
    :param weight: Weight Tensor
    :param bias: Bias Tensor
    :return: out = Ax [+ b]
    """
    # print("Input:{}".format(type(inputs)))
    # print("W:{}".format(type(weight)))
    # print("Bias:{}".format(type(bias)))
    if bias is None:
        return torch.mm(inputs, weight)
    else:
        return torch.addmm(bias, inputs, weight)


def decay_weight(weight_data):
    """
    Decay weights for nn layers
    :param weight_data: Tensor
    :return: Decayed weight Tensor
    """
    # TODO: Get decay rate from config
    return weight_data * 0.01


def relu(inputs):
    """
    Relu activation for input features
    :param inputs: Input features Tensor [2D]
    :return: ReLU activated Tensor [2D]
    """
    relu_activations = torch.clamp(inputs, min=0)
    if __debug__:
        if __dlevel__ is 2:
            print(inputs)
            print(relu_activations)
    return relu_activations


def softmax(inputs):
    exp_inputs = torch.exp(inputs)
    softmaxed = exp_inputs / torch.sum(exp_inputs, dim=1, keepdim=True)
    return softmaxed


def cross_entropy(inputs, targets):
    """
    :param inputs: softmaxed probabs
    :param targets: target indices list
    :return correct_log_probs: 
        (correct) negative log probabs 
        of targets only (list)
    """
    probs = correct_probs(inputs, targets)
    correct_log_probs = neg_log_probs(probs)
    return correct_log_probs


def correct_probs(inputs, targets):
    """
    :param inputs: softmaxed probabs
    :param targets: target indices list
    :return probabs of targets only (list) 
    """
    return inputs[range(len(inputs)), targets]


def neg_log_probs(inputs):
    """
    :param inputs: correct probabs
    :return negative log(base10) probabs
    """
    return -(log10(inputs))


def log10(inputs):
    """
    :param inputs: correct probabs
    :return log(base10) probabs
    """
    return torch.log(inputs) / torch.log(torch.Tensor([10]))


def average_loss(inputs):
    """
    :param inputs: losses list
    :return loss averaged over the list
    """
    return torch.sum(inputs) / len(inputs)


def gradient_softmax(inputs, targets):
    d_probs = inputs
    d_probs[range(len(inputs)), targets] -= 1
    d_probs /= len(inputs)
    return d_probs


def gradient_relu(activations):
    d_activations = activations
    d_activations[activations <= 0] = 0
    d_activations[activations == 0] = random.randint(1, 10) / 10.0
    return d_activations


def gradient_linear(gradients_output, weight):
    return torch.mm(gradients_output, weight.t())


def gradient_weight(inputs, gradient_output):
    return torch.mm(inputs.t(), gradient_output)


def gradient_bias(gradient_output):
    return torch.sum(gradient_output, dim=1, keepdim=True)
