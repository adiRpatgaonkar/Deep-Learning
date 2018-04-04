import torch


def flatten(data):
    """
    Flatten data Tensor
    :param data: 2D Tensor
    :param num_examples: Num of data points
    :return: flattened Tensor(1D Tensor)
    """
    num_examples = 1
    if data.dim() == 3:
        num_examples = 1
        data.resize(1, data.size())
    elif data.dim() == 4:
        num_examples = data.size(0)

    return data.view(num_examples, -1)


def standardize(data):
    """
    Standardize the given data with
    mean and standard deviation
    :param data: 2D Tensor(s)
    :return: Standardized data
    """
    data = flatten(data)
    mean = torch.mean(data, 1, keepdim=True)
    std_deviation = torch.std(data, 1, keepdim=True)
    data = data - mean
    data = data / std_deviation
    return data


def linear(inputs, weight, bias=None):
    """
    out = Ax + b
    :param inputs: Input features Tensor
    :param weight: Weight Tensor
    :param bias: Bias Tensor
    :return: out = Ax [+ b]
    """
    if bias is None:
        return torch.mm(inputs, weight)
    else:
        return torch.addmm(bias, inputs, weight)


def relu(inputs):
    """
    Relu activation for input features
    :param inputs: Input features Tensor
    :return: ReLU activated Tensor
    """
    relu_activations = torch.clamp(inputs, min=0)
    if __debug__:
        print(inputs)
        print(relu_activations)
    return relu_activations
