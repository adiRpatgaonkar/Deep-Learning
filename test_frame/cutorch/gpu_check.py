from __future__ import print_function

import torch


def available():
    global used
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # print("Using GPU")
        #print("GPU used: {}".format(torch.cuda.current_device()))
        used = True
        return True
    else:
        # print("Using cpu.")
        used = False
        return False
