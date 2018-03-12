" Setup before running loading/fiiting/training/testing models "

import torch
from termcolor import colored
import do_stuff as do


def setup_hardware():
    global dtype
    if do.use_gpu: # Want GPU
        if (-1 < do.args.GPU < torch.cuda.device_count()) and torch.cuda.is_available(): # GPU_ID exists & available
            # Subject to change
            torch.cuda.set_device(do.args.GPU) 
            print('\nUsing GPU: %d' % torch.cuda.current_device())
            print(colored('Check the GPU being used via "nvidia-smi" to avoid trouble mate!', 'red'))
            dtype = torch.cuda.FloatTensor
        else: # GPU_ID doesn't/isn't exist/available
            print("Selected GPU %d is not available." % do.args.GPU)
            do.use_gpu = False
            print('\nUsing CPU.')
            dtype = torch.FloatTensor
    else: # Want CPU
        print('\nUsing CPU.')
        dtype = torch.FloatTensor
