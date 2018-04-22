from __future__ import print_function

import math

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


def decay_weight(weight_data):
    """
    Decay weights for nn layers
    :param weight_data: Tensor
    :return: Decayed weight Tensor
    """
    # TODO: Get decay rate from config
    return weight_data * 0.01


#################################################
#                                               #
#                   FORWARD                     #
#                                               #
#################################################


def linear(inputs, weight, bias=None):
    """
    out = Ax + b
    :param inputs: Input features Tensor
    :param weight: Weight Tensor
    :param bias: Bias Tensor
    :return: out = Ax [+ b]
    """
    # print("Input:{}".format(inputs.size()))
    # print("W:{}".format(weight.size()))
    # print("Bias:{}".format(bias.size()))
    if bias is None:
        return torch.mm(inputs, weight)
    else:
        return torch.mm(inputs, weight) + bias


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


#################################################
#                                               #
#                     LOSS                      #
#                                               #
#################################################

def cross_entropy(inputs, targets):
    """
    :param inputs: softmaxed probs
    :param targets: target indices list
    :return correct_log_probs: 
        (correct) negative log probs
        of targets only (list)
    """
    probs = correct_probs(inputs, targets)
    negative_log_probs = neg_log_probs(probs)
    return negative_log_probs


def average_loss(inputs):
    """
    :param inputs: losses list
    :return loss averaged over the list
    """
    return torch.sum(inputs) / len(inputs)


def l1_regularization(strength, parameters):
    reg_loss = 0
    # print(strength, modules.values())

    for param_group in parameters:
        for param in param_group:
            if param.tag == 'weight':
                reg_loss += torch.sum(param.data*param.data)
    reg_loss *= (strength * 0.5)
    return reg_loss

#################################################
#                                               #
#                     Math                      #
#                                               #
#################################################


def correct_probs(inputs, targets):
    """
    :param inputs: softmaxed probs
    :param targets: target indices list
    :return probs of targets only (list)
    """
    return inputs[range(len(inputs)), targets]


def neg_log_probs(inputs):
    """
    :param inputs: correct probs
    :return negative log(base10) probs
    """
    return -(log10(inputs))


def log10(inputs):
    """
    :param inputs: correct probs
    :return log(base10) probs
    """
    return torch.log(inputs) / torch.log(torch.Tensor([10]))


def nan_check(data):
    if math.isnan(data):
        return True


#################################################
#                                               #
#                   GRADIENTS                   #
#                                               #
#################################################


def gradient_linear(weight, gradient_output):
    return torch.mm(gradient_output, weight.t())


def gradient_weight(inputs, gradient_output):
    return torch.mm(inputs.t(), gradient_output)


def gradient_bias(gradient_output):
    # TODO: Find out WHY dim = 0 ?
    return torch.sum(gradient_output, dim=0, keepdim=True)


def gradient_relu(activations, gradients):
    # Commented out. Leading to a nan loss
    # gradients[activations == 0] = random.randint(1, 10) / 10.0
    gradients[activations <= 0] = 0
    return gradients


def gradient_softmax(inputs, targets):
    d_probs = inputs
    d_probs[range(len(inputs)), targets] -= 1
    d_probs /= len(inputs)
    return d_probs
