from __future__ import print_function

import math

import torch
import numpy as np


def flatten(data):
    """
    Flatten data Tensor
    :param data: 2D Tensor
    :return: flattened Tensor(1D Tensor)
    """
    num_examples = 1
    if data.dim() == 3:
        num_examples = 1
        print("3D tensor given. 4D tensor preferred. Unsqueezing ... ")
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
#                     CONV                      #
#                                               #
#################################################

def conv2d(in_features, weight, bias=None):
    # Use post im2col op
    if bias is None:
        return torch.mm(weight, in_features)
    else:
        return torch.mm(weight, in_features) + bias
    

def max_pool2d(in_features):
    # Use post im2col op
    return torch.max(in_features, 1)

def im2col(image, kernel_size, stride):
    #print("im2col_in", image.size())
    im2col_out = torch.FloatTensor()
    fh = fw = kernel_size
    for i in range(0, image.size(1) - kernel_size + 1, stride):
        for j in range(0, image.size(2) - kernel_size + 1, stride):
            col_im = image[:, i:fh, j:fw]
            col_im = col_im.contiguous()  # Need to make tensor contiguous to flatten it
            col_im.unsqueeze_(0)
            col_im = col_im.view(col_im.size(0), -1)
            im2col_out = torch.cat((im2col_out, col_im.t()), 1)  # Cat. as col vector
            fw += stride  
        fh += stride
        fw = kernel_size  # Reset kernel width (Done parsing the width (j) for a certain i)
    fh = kernel_size  # Reset kernel height (Done parsing the height (i))
    #print("im2col_out", im2col_out.size())
    return im2col_out

def pad_image(image, padding):
    in_features = np.pad(image,
                         mode='constant', constant_values=0,
                         pad_width=((0, 0), (0, 0), 
                         (padding, padding), (padding, padding)))
    return in_features

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
    #print("I:", inputs)
    probs = inputs[range(len(inputs)), targets]
    #print("O:", probs)
    negative_log_probs = neg_log_probs(probs)
    return negative_log_probs


def average_loss(inputs):
    """
    :param inputs: losses list
    :return loss averaged over the list
    """
    return torch.sum(inputs) / len(inputs)


def l2_reg(strength, parameters):
    reg_loss = 0
    for param_group in parameters.values():
        for param in param_group:
            if param.tag == 'weight':
                reg_loss += torch.sum(param.data*param.data)
    reg_loss *= (strength * 0.5)
    return reg_loss

#################################################
#                                               #
#                     MATH                      #
#                                               #
#################################################

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
