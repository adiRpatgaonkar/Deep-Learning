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
    #print("Input:{}".format(type(inputs)))
    #print("W:{}".format(type(weight)))
    #print("Bias:{}".format(type(bias)))
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
    softmaxed = torch.exp(inputs) / torch.sum(inputs, dim=1, keepdim=True)
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
    correct_log_probs =  neg_log_probs(probs)
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
    return (-(log10(inputs)))

def log10(inputs):
    """
    :param inputs: correct probabs
    :return log(base10) probabs
    """
    return (torch.log(inputs) / torch.log(torch.Tensor([10])))

def average_loss(inputs):
    """
    :param inputs: losses list
    :param loss averaged over the list
    """
    return torch.sum(inputs) / len(inputs) 