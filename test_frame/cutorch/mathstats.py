import torch
import nn.functionals as f


def standardize(data):
    """
    Standardize the given data with
    mean and standard deviation
    :param data: 2D Tensor(s)
    :return: Standardized data
    """
    if data.dim() > 2:
        data = f.flatten(data)
    mean = torch.mean(data, 1, keepdim=True)
    std_deviation = torch.std(data, 1, keepdim=True)
    data = data - mean
    data = data / std_deviation
    return data
