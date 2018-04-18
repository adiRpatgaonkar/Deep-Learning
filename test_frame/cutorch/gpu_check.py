from __future__ import print_function

import torch


def available():
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        print("GPU used: {}".format(torch.cuda.current_device()))
        return True
    else:
        print("Using CPU.")
        return False