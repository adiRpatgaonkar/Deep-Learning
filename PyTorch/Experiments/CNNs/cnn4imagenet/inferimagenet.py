from matplotlib import pyplot as plt
import numpy as np
import torch


def see(input, mean=None, std=None, title=None):
    # DEBUG
    # Visualize a tensor
    assert torch.is_tensor(input), \
    "Expected tensor input. Got {}".format(type(input))
    assert input.dim() in (3, 4), \
    "Expected 3D or 4D tensor. Got {}D".format(input.dim())
    
    if input.is_cuda:
        input = input.cpu()

    if input.dim() == 3:
        input = input.unsqueeze(0)
    
    for inp in input:
        inp = inp.numpy().transpose((1, 2, 0))

        if mean and std:
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        if title:
            plt.title(title)
        plt.imshow(inp)
        plt.show()
        #plt.pause(0.001)