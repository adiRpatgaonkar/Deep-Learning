""" Setup before running loading/fitting/training/testing models """

# System imports
import torch
from termcolor import colored

# Custom imports
from libs.check_args import arguments, using_gpu


def setup_hardware():
    global dtype

    dtype = torch.FloatTensor
    args = arguments()
    use_gpu = using_gpu()
    if use_gpu:  # Want GPU
        if (-1 < args.GPU_ID < torch.cuda.device_count()) and torch.cuda.is_available():  # GPU_ID exists & available
            # Subject to change
            torch.cuda.set_device(args.GPU_ID)
            print('\nUSING GPU: %d' % torch.cuda.current_device())
            print(colored('Check the GPU being used via "nvidia-smi" to avoid trouble mate!', 'red'))
            dtype = torch.cuda.FloatTensor
        else:  # GPU_ID doesn't/isn't exist/available
            print("Selected GPU %d is NOT available." % args.GPU_ID)
            print('\nUSING CPU.')
            dtype = torch.FloatTensor
    else:  # Want CPU
        print('\nUSING CPU.')
        dtype = torch.FloatTensor


def default_tensor_type():
    return dtype
