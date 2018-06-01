from matplotlib import pyplot as plt
import numpy as np
import torch

classes = ('airplane', 'automobile',
           'bird', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck')
def see(inputs, mean=None, std=None, title=None):
    # DEBUG
    # Visualize a tensor
    assert torch.is_tensor(inputs), \
    "Expected tensor input. Got {}".format(type(inputs))
    assert inputs.dim() in (2, 3, 4), \
    "Expected 3D or 4D tensor. Got {}D".format(inputs.dim())
    
    if inputs.is_cuda:
        inputs = inputs.cpu()

    if inputs.dim() in (2, 3):
        inputs = inputs.unsqueeze(0)
    
    for inp in inputs:
        inp = inp.numpy()

        if inp.shape[0] == 3:
            inp = inp.transpose((1, 2, 0))

        if mean and std:
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

        if title:
            plt.title(title)
        print(inp, inp.shape)
        plt.imshow(inp, cmap="gray", interpolation="nearest")
        plt.show()
        #plt.pause(0.001)
        