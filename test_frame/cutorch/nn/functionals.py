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


#################################################
#                                               #
#                   CONV-OPS                    #
#                                               #
#################################################

def im2col(image, kernel_size, stride, task="conv"):
    # One image @ a time
    im2col_out = torch.FloatTensor()
    # To parse across width and height (temp vars).
    # Keep kernel_size constant
    fh = fw = kernel_size 
    for i in range(0, image.size(1) - kernel_size + 1, stride):
        for j in range(0, image.size(2) - kernel_size + 1, stride):
            im_col = image[:, i:fh, j:fw]
            im_col = im_col.contiguous()  # Need to make tensor contiguous to flatten it
            im_col.unsqueeze_(0) # Stretch to 4D tensor
            if task == "conv":
                # Flatten across 3D space
                im_col = im_col.view(im_col.size(0), -1)
            elif task == "pooling": 
                # Flatten across 2D i.e. preserve depth dim
                im_col = im_col.view(im_col.size(1), -1)
            im2col_out = torch.cat((im2col_out, im_col.t()), 1)  # Cat. as col vector
            fw += stride  
        fh += stride
        fw = kernel_size  # Reset kernel width (Done parsing the width (j) for a certain i)
    fh = kernel_size  # Reset kernel height (Done parsing the height (i))
    # Clean
    del image, im_col
    return im2col_out


def pad_image(image, p):
    # Works for a batch of images
    # i.e. 4D tensor
    if torch.is_tensor(image):
        if image.dim() != 4:
            raise ValueError("Works for 4D Tensor only.")
        image = image.numpy()
    elif type(image) != np.ndarray:
        raise TypeError("Padding will be done on a numpy array only.")
    image = np.pad(image, mode='constant', constant_values=0,
                   pad_width=((0,0), (0,0), (p,p), (p,p)))
    return image

#################################################
#                                               #
#                   FORWARD                     #
#                                               #
#################################################

def batchnorm_2d(x, beta, gamma, epsilon):
    assert x.dim() == 4, ("Input should be a 4D Tensor")
    N, C, H, W = x.size()
    mean, variance = [], []
    # Mean: w.r.t channels
    for channel in range(C):
        mean.append(torch.mean(x[:, channel, :, :]))
    mean = torch.Tensor(mean)
    # print(mean)
    # Variance: w.r.t channels
    x_mu = (x - mean.view(1, C, 1, 1))
    x_mu_sq = x_mu ** 2
    for channel in range(C):
        variance.append(torch.mean(x_mu_sq[:, channel, :, :]))
    variance = torch.Tensor(variance)  # sigma^2
    # print(variance)
    sqrt_var = torch.sqrt(variance.view(1, C, 1, 1) + epsilon)
    invr_var = 1.0 / sqrt_var
    # Normalized input.
    x_hat = x_mu * invr_var
    # print(x_hat)
    # Output. Scale & shift.
    out = gamma.view(1, C, 1, 1) * x_hat + beta.view(1, C, 1, 1)
    # print(out)
    cache = (x_hat, invr_var) # To be used for backwarding BN2D
    del x, x_mu, x_mu_sq, sqrt_var
    # Clean
    return out, mean, variance, cache 


def conv_2d(input, weight, bias=None):
    # Use post im2col op
 
    # Batch matrix multiplication. Adam Paszke's solution in Pytorch forums
    # Multiplying 3D input tensor with 2D weight tensor
    if input.dim() == 3: # 3D in_features tensor 
        weight = weight.unsqueeze(0).expand(input.size(0), *weight.size())
        if bias is None:  
            return torch.bmm(weight, input)
        else:    
            return torch.bmm(weight, input) + bias.unsqueeze(0).expand(input.size(0), *bias.size())

    # Matrix multiplication. input: 2D tensor, weight: 2D tensor
    else:
        if  bias is None:
            return torch.mm(weight, input)
        else:
            return torch.mm(weight, input) + bias
    

def max_pool2d(input):
    # Use post im2col op
    return torch.max(input, 1)


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
    # relu_activations = torch.clamp(inputs, min=0)
    return torch.clamp(inputs, min=0)


def softmax(inputs):
    exp_inputs = torch.exp(inputs)
    # softmaxed = exp_inputs / torch.sum(exp_inputs, dim=1, keepdim=True)
    return exp_inputs / torch.sum(exp_inputs, dim=1, keepdim=True) 

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
    # negative_log_probs = neg_log_probs(probs)
    return neg_log_probs(probs)


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
    # Clean
    del activations
    return gradients


def gradient_softmax(inputs, targets):
    d_probs = inputs
    d_probs[range(len(inputs)), targets] -= 1
    d_probs /= len(inputs)
    # Clean
    del inputs
    return d_probs


def gradient_beta(grad_output):
    #print("dout:", grad_output)
    grad_beta = []
    for c in range(grad_output.size(1)):
        grad_beta.append(torch.sum(grad_output[:, c, :, :]))
    return torch.Tensor(grad_beta)


def gradient_gamma(x_hat, grad_output):
    #print("dout:", grad_output)
    grad_gamma = []
    for c in range(grad_output.size(1)):
        grad_gamma.append(torch.sum(grad_output[:, c, :, :]))
    return torch.Tensor(grad_gamma)


def gradient_bnorm2d(gamma, cache, grad_output):
    N, C, H, W = grad_output.size()
    x_hat, invr_var = cache
    grad_xhat = grad_output.clone()
    sum_grad_xhat = []
    sum_xhat_grad_xhat = []
    for c in range(C):
        grad_xhat[:, c, :, :] *= gamma[c]  # Intermediate gradient of x_hat 
        sum_grad_xhat.append(torch.sum(grad_xhat[:, c, :, :]))
        sum_xhat_grad_xhat.append(torch.sum(x_hat[:, c, :, :] * grad_xhat[:, c, :, :]))
    grad_xhat = torch.Tensor(grad_xhat)  # Intermediate term 1
    sum_grad_xhat = torch.Tensor(sum_grad_xhat)  # Intermediate term 2 
    sum_xhat_grad_xhat = torch.Tensor(sum_xhat_grad_xhat) # Intermediate term 3
    # Gradient input: Final expression
    grad_in = (1. / N) * invr_var * ((N * grad_xhat) - sum_grad_xhat.view(1, C, 1, 1) - (x_hat * sum_xhat_grad_xhat.view(1, C, 1, 1)))
    # Clean
    del gamma, cache, grad_output, x_hat, invr_var, grad_xhat, sum_grad_xhat, sum_xhat_grad_xhat
    return grad_in

