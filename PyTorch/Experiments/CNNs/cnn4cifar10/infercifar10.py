from matplotlib import pyplot as plt
import numpy as np
import torch

classes = ('airplane', 'automobile',
           'bird', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck')
def see(input, mean=None, std=None, title=None):
    # DEBUG
    print(input, input.size(), input.dtype)
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
            plt.title(classes[title])
        plt.imshow(inp)
        plt.show()
        #plt.pause(0.001)