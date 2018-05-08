from __future__ import print_function

import math

import torch
import numpy as np


def flatten(data):
    # Maybe redundant & hence deprecated.
    """
    Flatten data Tensor
    :param data: 2D Tensor
    :return: flattened Tensor(1D Tensor)
    """
    assert data.dim() in (3, 4), "Expected 3D or 4D tensor, got {}D tensor".format(data.dim)
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

def im2col(batch, kernel_size, stride):
    """ im2col for batch inputs i.e. 4D Tensors """
    assert batch.dim() == 4, "Input batch should be 4D tensor"
    batch_i2c = torch.Tensor()
    # To parse across width and height (temp vars).
    # Keep kernel_size constant
    fh = fw = kernel_size
    for image in batch:
        i2c_per_im = torch.Tensor()
        for i in range(0, image.size(1) - kernel_size + 1, stride):
            for j in range(0, image.size(2) - kernel_size + 1, stride):
                im_col = image[:, i:fh, j:fw]
                im_col = im_col.contiguous()  # tensor must be contiguous to flatten it
                im_col.unsqueeze_(0)  # Stretch to 4D tensor
                im_col = im_col.view(im_col.size(0), -1)
                i2c_per_im = torch.cat((i2c_per_im, im_col.t()), 1)  # Cat as col vector
                fw += stride
            fh += stride
            fw = kernel_size  # Reset kernel width
        fh = kernel_size  # Reset kernel height
        batch_i2c = torch.cat((batch_i2c, i2c_per_im), 1)
        # print("Bim2c",batch_im2col)
    # Clean
    del batch, i2c_per_im, im_col
    return batch_i2c


def col2im(input, x_size, k_size, stride):
    """ col2im or de-im2col for 2D tensors """
    assert input.dim() == 2, "Input should be a 2D tensor."
    assert len(x_size) == 4, "Input size should have 4 values for 4D tensor."
    batch_c2i = torch.Tensor()
    N, C, H, W = x_size
    batch = torch.chunk(input, N, dim=1)
    # To parse across width and height (temp vars).
    # Keep kernel_size constant
    fh = fw = k_size
    for image in batch:
        m = 0  # Number of locations convolved(mth col)

        c2i_per_im = torch.zeros(C, H, W)
        for i in range(0, H - k_size + 1, stride):
            for j in range(0, W - k_size + 1, stride):
                col = image[:, m].contiguous()
                c2i_per_im[:, i:fh, j:fw] = col.view(C, k_size, k_size)
                m += 1  #
                fw += stride
            fh += stride
            fw = k_size
        fh = k_size
        batch_c2i = torch.cat((batch_c2i, c2i_per_im.unsqueeze(0)), 0)
    del input, batch, col, c2i_per_im
    return batch_c2i


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
                   pad_width=((0, 0), (0, 0), (p, p), (p, p)))
    return image


#################################################
#                                               #
#                   FORWARD                     #
#                                               #
#################################################

def batchnorm_2d(x, beta, gamma, epsilon, mean=None, var=None):
    assert x.dim() == 4, "Input should be a 4D Tensor"
    N, C, H, W = x.size()
    if not mean and not var:
        # Training
        mean, variance = [], []
        # Mean: w.r.t channels
        for channel in range(C):
            mean.append(torch.mean(x[:, channel, :, :]))
        mean = torch.Tensor(mean) # New mean
        
        # print(mean)
        # Variance: w.r.t channels
        x_mu = (x - mean.view(1, C, 1, 1))
        x_mu_sq = x_mu ** 2
        for channel in range(C):
            variance.append(torch.mean(x_mu_sq[:, channel, :, :]))
        variance = torch.Tensor(variance)  # sigma^2 # New var
    # else use running mean as mean and var is given
    x_mu = (x - mean.view(1, C, 1, 1))
    x_mu_sq = x_mu ** 2 
    sqrt_var = torch.sqrt(variance.view(1, C, 1, 1) + epsilon)
    invr_var = 1.0 / sqrt_var
    # Normalized input.
    x_hat = x_mu * invr_var
    # print(x_hat)
    # Output. Scale & shift.
    out = gamma.view(1, C, 1, 1) * x_hat + beta.view(1, C, 1, 1)
    # print(out)
    cache = (x_hat, invr_var)  # To be used for backwarding BN2D
    del x, x_mu, x_mu_sq, sqrt_var
    # Clean
    return out, mean, variance, cache


def conv_2d(input, weight, bias=None):
    # Use post im2col op 
    if bias is None:
        return torch.mm(weight, input)
    else:
        return torch.mm(weight, input) + bias


def max_pool2d(input):
    # Use post im2col op
    return torch.max(input, 0)


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
    # print("I:", inputs)
    probs = inputs[range(len(inputs)), targets]
    # print("O:", probs)
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
                reg_loss += torch.sum(param.data * param.data)
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


def gradient_beta(grad_out):
    # print("dout:", grad_out)
    grad_beta = []
    for c in range(grad_out.size(1)):
        grad_beta.append(torch.sum(grad_out[:, c, :, :]))
    return torch.Tensor(grad_beta)


def gradient_gamma(grad_out):
    # print("dout:", grad_out)
    grad_gamma = []
    for c in range(grad_out.size(1)):
        grad_gamma.append(torch.sum(grad_out[:, c, :, :]))
    return torch.Tensor(grad_gamma)


def gradient_bnorm2d(gamma, cache, grad_out):
    N, C, H, W = grad_out.size()
    x_hat, invr_var = cache
    grad_xhat = grad_out.clone()
    sum_grad_xhat = []
    sum_xhat_grad_xhat = []
    for c in range(C):
        grad_xhat[:, c, :, :] *= gamma[c]  # Intermediate gradient of x_hat 
        sum_grad_xhat.append(torch.sum(grad_xhat[:, c, :, :]))
        sum_xhat_grad_xhat.append(torch.sum(x_hat[:, c, :, :] * grad_xhat[:, c, :, :]))
    grad_xhat = torch.Tensor(grad_xhat)  # Intermediate term 1
    sum_grad_xhat = torch.Tensor(sum_grad_xhat)  # Intermediate term 2 
    sum_xhat_grad_xhat = torch.Tensor(sum_xhat_grad_xhat)  # Intermediate term 3
    # Gradient input: Final expression
    grad_in = (1. / N) * invr_var * (
                (N * grad_xhat) - sum_grad_xhat.view(1, C, 1, 1) - (x_hat * sum_xhat_grad_xhat.view(1, C, 1, 1)))
    # Clean
    del gamma, cache, grad_out, x_hat, invr_var, grad_xhat, sum_grad_xhat, sum_xhat_grad_xhat
    return grad_in


def grad_conv2d_bias(grad_out):
    C = grad_out.size(1)
    grad_bias = []

    for c in range(C):
        grad_bias.append(torch.sum(grad_out[:, c, :, :]))
    # Clean
    del grad_out
    return torch.Tensor(grad_bias)


def grad_conv2d_weight(input, grad_out):
    C = grad_out.size(1)
    grad_out = grad_out.permute(1, 2, 3, 0).contiguous()
    grad_out = grad_out.view(C, -1)
    grad_weight = torch.mm(grad_out, input.t())
    # Clean
    del input
    return grad_weight, grad_out


def grad_conv2d(cache, weight, grad_out):
    C = grad_out.size(1)
    weight_reshaped = weight.view(C, -1)
    grad_Xcol = torch.mm(weight_reshaped.t(), cache)
    # Clean
    del cache, weight_reshaped, grad_out
    return grad_Xcol


def grad_maxpool2d(xcol_size, max_track, grad_out):
    rows, cols = xcol_size
    grad_Xcol = torch.Tensor(rows, cols).fill_(0)
    grad_out = grad_out.permute(2, 3, 0, 1).contiguous()
    grad_out = grad_out.view(-1)
    grad_Xcol[max_track, range(max_track.size(0))] = grad_out
    # Clean
    del grad_out
    return grad_Xcol
